#!/usr/bin/env python3
"""Direct BIRD smoke runner for the Python service without starting FastAPI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BACKEND = REPO_ROOT / "python_backend"
if str(PYTHON_BACKEND) not in sys.path:
    sys.path.insert(0, str(PYTHON_BACKEND))

from app.bird_eval import BirdEvalRequest, BirdEvalService  # noqa: E402
from app.config import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Python BIRD smoke evaluation")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "bird" / "dev_20240627"),
    )
    parser.add_argument("--question-ids", required=True, help="comma/space separated ids")
    parser.add_argument("--mode", choices=("agent", "baseline"), default="agent")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=2)
    return parser.parse_args()


def parse_ids(value: str) -> list[str]:
    return [
        item
        for token in value.replace(",", " ").split()
        if (item := token.strip())
    ]


def find_db_root(data_root: Path) -> Path:
    root = data_root / "dev_databases"
    if (root / "dev_databases").is_dir():
        root = root / "dev_databases"
    return root


def execute_rows(path: Path, sql: str) -> tuple[bool, set[tuple[Any, ...]], str]:
    if not sql.strip():
        return False, set(), "EMPTY_SQL"
    connection: sqlite3.Connection | None = None
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        rows = set(tuple(row) for row in connection.execute(sql).fetchall())
        return True, rows, ""
    except Exception as exc:
        return False, set(), f"{type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()


async def main_async(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    dataset = json.loads((data_root / "dev.json").read_text(encoding="utf-8"))
    by_id = {str(row["question_id"]): row for row in dataset}
    question_ids = parse_ids(args.question_ids)
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise SystemExit("Unknown question_ids: " + ", ".join(missing))

    settings = Settings(
        _env_file=PYTHON_BACKEND / ".env",
        bird_eval_enabled=True,
    )
    service = BirdEvalService(settings)
    db_root = find_db_root(data_root)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def run_one(qid: str) -> dict[str, Any]:
        row = by_id[qid]
        db_id = str(row["db_id"])
        db_path = db_root / db_id / f"{db_id}.sqlite"
        req = BirdEvalRequest(
            questionId=qid,
            dbId=db_id,
            question=str(row["question"]),
            evidence=str(row.get("evidence") or ""),
            sqlitePath=str(db_path),
            maxRounds=args.max_rounds if args.mode == "agent" else None,
        )
        async with semaphore:
            if args.mode == "agent":
                response = await service.eval_agent(req)
            else:
                response = await service.eval_baseline(req)

        pred_sql = str(response.get("sql") or "")
        gold_sql = str(row.get("SQL") or "")
        pred_ok, pred_rows, pred_error = execute_rows(db_path, pred_sql)
        gold_ok, gold_rows, gold_error = execute_rows(db_path, gold_sql)
        correct = pred_ok and gold_ok and pred_rows == gold_rows
        return {
            "question_id": qid,
            "db_id": db_id,
            "success": bool(response.get("success")),
            "correct": correct,
            "sql": pred_sql,
            "error": response.get("error") or pred_error or gold_error,
            "attempts": response.get("attempts"),
            "toolRounds": response.get("toolRounds"),
            "probeCalls": response.get("probeCalls"),
            "requestedColumns": response.get("requestedColumns") or [],
            "toolHistory": response.get("toolHistory") or [],
        }

    results = await asyncio.gather(*(run_one(qid) for qid in question_ids))
    for item in results:
        print(json.dumps(item, ensure_ascii=False))
    correct = sum(int(item["correct"]) for item in results)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "correct": correct,
                "total": len(results),
                "ex": round(correct * 100.0 / len(results), 2) if results else 0.0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if correct == len(results) else 3


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
