#!/usr/bin/env python3
"""BIRD Dev evaluation runner for dodo-agentx."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REGRESSION_MANIFEST = SCRIPT_DIR / "report" / "bird_java_correct_python_wrong.json"
MAX_BLANK_SQL_RETRIES = 3
TRANSIENT_GENERATION_ERROR_MARKERS = (
    "upstream stream ended before terminal chunk",
    "stream ended before terminal chunk",
    "connection reset",
    "server disconnected",
)
WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class SqlResult:
    ok: bool
    rows: set[tuple]
    row_count: int
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIRD-SQL Dev local evaluation runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8889",
                        help="DataAna Python service address")
    parser.add_argument("--mode", choices=("agent", "baseline"), default="agent",
                        help="Python LangGraph ReAct/tool-loop or direct ChatModel baseline")
    parser.add_argument("--endpoint", default="",
                        help="override evaluation API path; defaults from --mode")
    parser.add_argument("--data-root", required=True,
                        help="BIRD dev 数据包解压后的根目录，需包含 dev.json 和 dev_databases")
    parser.add_argument("--output", default="",
                        help="prediction JSONL path; defaults to report/pred.python.<mode>.jsonl")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="evaluate the first N selected questions; 0 means all")
    parser.add_argument("--question-ids", default="",
                        help="comma/space separated question_ids to evaluate")
    parser.add_argument("--regression-only", action="store_true",
                        help="evaluate Java-agent-correct/Python-agent-wrong questions derived from reports")
    parser.add_argument("--java-report", default=str(SCRIPT_DIR / "report" / "pred.report.md"),
                        help="historical Java agent report used by --regression-only")
    parser.add_argument("--python-report", default=str(SCRIPT_DIR / "report" / "pred.python.agent.report.md"),
                        help="Python agent report used to derive the historical regression set")
    parser.add_argument("--regression-manifest", default=str(DEFAULT_REGRESSION_MANIFEST),
                        help="frozen Java-correct/Python-wrong regression manifest")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="number of questions requested concurrently")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="HTTP timeout in seconds")
    parser.add_argument("--retry", type=int, default=2,
                        help="HTTP retry count with exponential backoff")
    parser.add_argument("--generation-retry", type=int, default=2,
                        help="retry whole questions when the API returns a transient model-stream failure")
    parser.add_argument("--sql-timeout", type=float, default=300.0,
                        help="SQL execution timeout in seconds; slow gold SQL needs headroom")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="rerun failed questions from the prediction file")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="override ReAct/tool-loop model rounds per question; 0 uses service default")
    return parser.parse_args()


def parse_question_ids(value: str) -> set[str]:
    return {item for item in re.split(r"[\s,]+", value.strip()) if item}


def wrong_question_ids_from_report(path: Path) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"BIRD report not found: {path}")
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^### question_id=(\S+)", text, flags=re.MULTILINE))


def derive_regression_question_ids(java_report: Path, python_report: Path) -> list[str]:
    java_wrong = wrong_question_ids_from_report(java_report)
    python_wrong = wrong_question_ids_from_report(python_report)
    return sorted(python_wrong - java_wrong, key=question_sort_key)


def load_regression_manifest(path: Path) -> list[str]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("question_ids"), list):
        raise SystemExit(f"Invalid regression manifest: {path}")
    return sorted({str(value) for value in payload["question_ids"]}, key=question_sort_key)


def load_dataset(input_json: Path) -> list[dict[str, Any]]:
    rows = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid BIRD dataset, expected a JSON array: {input_json}")

    required = {"question_id", "db_id", "question", "evidence", "SQL", "difficulty"}
    dataset: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            missing = sorted(required - set(row if isinstance(row, dict) else {}))
            raise SystemExit(f"Invalid dataset row {index}, missing fields: {missing}")
        item = dict(row)
        item["question_id"] = str(item["question_id"])
        dataset.append(item)
    return dataset


def load_done(output_path: Path, only_success: bool) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("question_id") is None:
                continue
            if only_success and record.get("error"):
                continue
            done.add(str(record["question_id"]))
    return done


def load_predictions(output_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not output_path.exists():
        return records

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            question_id = str(record.get("question_id", ""))
            if question_id:
                records[question_id] = record
    return dict(sorted(records.items(), key=lambda item: question_sort_key(item[0])))


def question_sort_key(question_id: str) -> tuple[int, int] | tuple[int, str]:
    try:
        return (0, int(question_id))
    except (TypeError, ValueError):
        return (1, str(question_id))


def append_prediction(
    output_path: Path,
    question_id: str,
    db_id: str,
    sql: str,
    error: str = "",
    duration_ms: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "question_id": question_id,
        "db_id": db_id,
        "pred": sql,
        "duration_ms": duration_ms,
    }
    if error:
        record["error"] = error
    if diagnostics:
        record["diagnostics"] = diagnostics

    with WRITE_LOCK:
        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def eval_diagnostics(response: dict[str, Any]) -> dict[str, Any]:
    """Persist service-side Agent diagnostics without changing evaluation scoring."""
    keys = (
        "mode",
        "attempts",
        "approved",
        "critic",
        "probe",
        "toolRounds",
        "probeCalls",
        "semanticRepairs",
        "requestedColumns",
        "taskContract",
        "semanticVerification",
        "toolHistory",
    )
    return {key: response.get(key) for key in keys if key in response}


def sqlite_path(db_root: Path, db_id: str) -> Path:
    # The evaluation API runs from python_backend/, so payload paths must be absolute.
    # A relative path would be resolved against the service cwd instead of the runner cwd.
    return (db_root / db_id / f"{db_id}.sqlite").expanduser().resolve()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("evaluation API must return a JSON object")
    return result


def call_eval(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + args.endpoint
    last_error: Exception | None = None
    for attempt in range(args.retry + 1):
        try:
            return post_json(url, payload, args.timeout)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < args.retry:
                time.sleep(2**attempt)
    raise RuntimeError(f"evaluation API failed: {last_error}") from last_error


def is_transient_generation_error(error: str) -> bool:
    normalized = (error or "").strip().lower()
    return any(marker in normalized for marker in TRANSIENT_GENERATION_ERROR_MARKERS)


def process_question(row: dict[str, Any], args: argparse.Namespace,
                     output_path: Path) -> tuple[str, bool, int]:
    question_id = row["question_id"]
    db_id = str(row["db_id"])
    started = time.monotonic()
    database_path = sqlite_path(Path(args.db_root), db_id)
    if not database_path.is_file():
        error = f"SQLite database not found: {database_path}"
        duration_ms = int((time.monotonic() - started) * 1000)
        append_prediction(output_path, question_id, db_id, "", error, duration_ms)
        return question_id, False, duration_ms

    payload = {
        "questionId": question_id,
        "dbId": db_id,
        "question": str(row["question"]),
        "evidence": str(row["evidence"] or ""),
        "sqlitePath": str(database_path),
    }
    if args.max_rounds > 0:
        payload["maxRounds"] = args.max_rounds

    last_error = ""
    last_duration_ms = 0
    blank_sql_retries = 0
    transient_generation_retries = 0
    while True:
        try:
            response = call_eval(args, payload)
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            append_prediction(output_path, question_id, db_id, "", str(exc), duration_ms)
            return question_id, False, duration_ms

        sql = str(response.get("sql") or "").strip()
        success = bool(response.get("success"))
        last_error = "" if success else str(response.get("error") or "")
        last_duration_ms = int(response.get("durationMs") or 0)
        if success and sql:
            append_prediction(
                output_path,
                question_id,
                db_id,
                sql,
                "",
                last_duration_ms,
                eval_diagnostics(response),
            )
            return question_id, True, last_duration_ms

        if "blank SQL" in last_error and blank_sql_retries < MAX_BLANK_SQL_RETRIES:
            blank_sql_retries += 1
            time.sleep(1)
            continue
        if (
            is_transient_generation_error(last_error)
            and transient_generation_retries < max(0, int(args.generation_retry))
        ):
            transient_generation_retries += 1
            time.sleep(2 ** (transient_generation_retries - 1))
            continue
        break

    duration_ms = int((time.monotonic() - started) * 1000)
    diagnostics = eval_diagnostics(response) if "response" in locals() else None
    append_prediction(
        output_path,
        question_id,
        db_id,
        "",
        last_error,
        duration_ms,
        diagnostics,
    )
    return question_id, False, last_duration_ms or duration_ms


def execute_sql(database_path: Path, sql: str, timeout_seconds: float) -> SqlResult:
    if not sql.strip():
        return SqlResult(False, set(), 0, "EMPTY_SQL")
    if not database_path.is_file():
        return SqlResult(False, set(), 0, f"DATABASE_NOT_FOUND: {database_path}")

    connection = None
    timed_out = False
    try:
        database_uri = database_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(
            database_uri, uri=True, timeout=min(timeout_seconds, 30.0)
        )
        connection.execute("PRAGMA query_only=ON")
        deadline = time.monotonic() + timeout_seconds

        def interrupt_when_timeout() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        connection.set_progress_handler(interrupt_when_timeout, 1000)
        rows = connection.execute(sql).fetchall()
        return SqlResult(True, set(rows), len(rows), "")
    except Exception as exc:
        error = f"TIMEOUT after {timeout_seconds}s" if timed_out else f"{type(exc).__name__}: {exc}"
        return SqlResult(False, set(), 0, error)
    finally:
        if connection is not None:
            connection.close()


def evaluate_predictions(args: argparse.Namespace, dataset: list[dict[str, Any]],
                         output_path: Path) -> list[dict[str, Any]]:
    predictions = load_predictions(output_path)
    cases: list[dict[str, Any]] = []

    for row in dataset:
        question_id = row["question_id"]
        prediction = predictions.get(question_id)
        predicted_sql = str(prediction.get("pred") or "") if prediction else ""
        gold_sql = str(row.get("SQL") or "")
        db_id = str(row["db_id"])
        database_path = sqlite_path(Path(args.db_root), db_id)

        predicted = execute_sql(database_path, predicted_sql, args.sql_timeout)
        gold = execute_sql(database_path, gold_sql, args.sql_timeout)
        correct = predicted.ok and gold.ok and predicted.rows == gold.rows
        cases.append({
            "question_id": question_id,
            "db_id": db_id,
            "difficulty": str(row.get("difficulty") or ""),
            "question": str(row.get("question") or ""),
            "evidence": str(row.get("evidence") or ""),
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql,
            "generation_error": str(prediction.get("error") or "") if prediction else "MISSING_PREDICTION",
            "diagnostics": dict(prediction.get("diagnostics") or {}) if prediction else {},
            "predicted": predicted,
            "gold": gold,
            "correct": correct,
        })
    return cases


def score_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    levels = ("simple", "moderate", "challenging")
    summary: dict[str, dict[str, float | int]] = {}
    for level in levels + ("total",):
        selected = cases if level == "total" else [case for case in cases if case["difficulty"] == level]
        correct_count = sum(bool(case["correct"]) for case in selected)
        summary[level] = {
            "count": len(selected),
            "correct": correct_count,
            "ex": correct_count * 100.0 / len(selected) if selected else 0.0,
        }
    return summary


def table_text(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def sql_block(sql: str) -> str:
    return f"```sql\n{sql or '-- EMPTY SQL --'}\n```"


def format_duration(milliseconds: int) -> str:
    total_seconds = milliseconds / 1000
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes}m{seconds:02d}s"


def write_report(args: argparse.Namespace, cases: list[dict[str, Any]],
                 output_path: Path) -> Path:
    summary = score_summary(cases)
    predictions = load_predictions(output_path)
    report_path = output_path.with_suffix(".report.md")
    failed = [case for case in cases if case["generation_error"]]
    wrong = [case for case in cases if not case["correct"]]
    duration_ms = sum(
        int(predictions.get(case["question_id"], {}).get("duration_ms") or 0)
        for case in cases
    )

    lines = [
        "# BIRD 评测报告",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 数据目录：`{args.data_root}`",
        f"- 评测接口：`{args.endpoint}`",
        "",
        "## 分数",
        "",
        "| 难度 | 题数 | 正确题数 | EX |",
        "| --- | ---: | ---: | ---: |",
    ]
    for level in ("simple", "moderate", "challenging", "total"):
        item = summary[level]
        lines.append(f"| {level} | {item['count']} | {item['correct']} | {item['ex']:.2f} |")

    lines.extend([
        "",
        "## 运行摘要",
        "",
        f"- 评分题数：{len(cases)}",
        f"- 生成失败：{len(failed)}",
        f"- 错题数量：{len(wrong)}",
    ])
    if duration_ms:
        lines.append(f"- 生成 SQL 累计耗时：{format_duration(duration_ms)}（并发执行，非墙钟时间）")

    lines.extend(["", "## 生成失败", ""])
    if failed:
        lines.extend(["| question_id | db_id | 失败原因 |", "| --- | --- | --- |"])
        for case in failed:
            lines.append(
                f"| {case['question_id']} | {case['db_id']} "
                f"| {table_text(case['generation_error'])} |"
            )
    else:
        lines.append("无。")

    lines.extend(["", "## 错题分析", ""])
    if not wrong:
        lines.append("无错题。")
    for case in wrong:
        predicted = case["predicted"]
        gold = case["gold"]
        lines.extend([
            f"### question_id={case['question_id']}",
            "",
            f"- 数据库：`{case['db_id']}`",
            f"- 难度：`{case['difficulty']}`",
            f"- 问题：{table_text(case['question'])}",
            f"- Evidence：{table_text(case['evidence'])}",
            "",
            "预测 SQL：",
            "",
            sql_block(case["predicted_sql"]),
            "",
            "标准 SQL：",
            "",
            sql_block(case["gold_sql"]),
            "",
        ])
        if predicted.error:
            lines.append(f"- 预测执行：`{table_text(predicted.error)}`")
        else:
            lines.append(f"- 预测执行：成功，{predicted.row_count} 行")
        if gold.error:
            lines.append(f"- 标准执行：`{table_text(gold.error)}`")
        else:
            lines.append(f"- 标准执行：成功，{gold.row_count} 行")
        diagnostics = case.get("diagnostics") or {}
        if diagnostics:
            lines.extend([
                "",
                "Agent trace：",
                "",
                "```json",
                json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str),
                "```",
            ])
        lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[eval] report: {report_path}")
    return report_path


def run(args: argparse.Namespace) -> int:
    if not args.endpoint:
        args.endpoint = "/bird/eval/baseline" if args.mode == "baseline" else "/bird/eval/question"
    if not args.output:
        suffix = ".regression" if args.regression_only else ""
        args.output = str(SCRIPT_DIR / "report" / f"pred.python.{args.mode}{suffix}.jsonl")

    data_root = Path(args.data_root)
    input_json = data_root / "dev.json"
    db_root = data_root / "dev_databases"
    if (db_root / "dev_databases").is_dir():
        db_root = db_root / "dev_databases"
    output_path = Path(args.output)
    if not input_json.is_file():
        raise SystemExit(f"--data-root 下找不到 dev.json：{input_json}，"
                         f"请确认指向解压后包含 dev.json 和 dev_databases 的目录")
    if not db_root.is_dir():
        raise SystemExit(f"--data-root 下找不到 dev_databases：{db_root}")
    args.db_root = str(db_root)

    dataset = load_dataset(input_json)
    selected = dataset

    requested_ids = parse_question_ids(args.question_ids)
    if requested_ids:
        selected = [row for row in selected if row["question_id"] in requested_ids]
        found = {row["question_id"] for row in selected}
        missing = sorted(requested_ids - found, key=question_sort_key)
        if missing:
            raise SystemExit("Unknown --question-ids: " + ", ".join(missing))

    if args.regression_only:
        regression_manifest = Path(args.regression_manifest)
        regression_values = load_regression_manifest(regression_manifest)
        source = str(regression_manifest)
        if not regression_values:
            regression_values = derive_regression_question_ids(
                Path(args.java_report), Path(args.python_report)
            )
            source = f"derived:{args.java_report} - {args.python_report}"
        regression_ids = set(regression_values)
        selected = [row for row in selected if row["question_id"] in regression_ids]
        print(f"[runner] regression_source={source}")
        print(
            "[runner] regression_ids="
            + ",".join(row["question_id"] for row in selected)
        )

    if args.max_questions > 0:
        selected = selected[: args.max_questions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output_path, only_success=args.rerun_failed)
    todo = [row for row in selected if row["question_id"] not in done]

    print(f"[runner] mode={args.mode} endpoint={args.endpoint}")
    print(f"[runner] dataset={input_json}")
    print(f"[runner] db-root={db_root}")
    print(f"[runner] selected={len(selected)} todo={len(todo)} skipped={len(selected) - len(todo)}")

    success = 0
    started = time.monotonic()
    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(process_question, row, args, output_path) for row in todo]
            for index, future in enumerate(as_completed(futures), start=1):
                question_id, ok, duration_ms = future.result()
                success += int(ok)
                status = "成功" if ok else "失败"
                print(f"[runner] [{index}/{len(todo)}] qid={question_id} {status} 耗时={duration_ms / 1000:.1f}s")

    cases = evaluate_predictions(args, selected, output_path)
    write_report(args, cases, output_path)
    elapsed = time.monotonic() - started
    print(f"[runner] 完成 | 生成成功={success} 生成失败={len(todo) - success} 总耗时={elapsed:.1f}s")
    return 0 if success == len(todo) else 2


if __name__ == "__main__":
    sys.exit(run(parse_args()))
