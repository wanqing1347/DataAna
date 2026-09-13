#!/usr/bin/env python3
"""Run the 120-question BIRD resume benchmark against the Python service."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXPECTED = {
    "baseline": {"correct": 70, "total": 120, "ex": 58.33},
    "agent": {"correct": 83, "total": 120, "ex": 69.17},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the BIRD resume 120-question scores")
    parser.add_argument("--data-root", default="", help="BIRD dev root; auto-detected when omitted")
    parser.add_argument("--base-url", default="http://127.0.0.1:8889")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="keep and resume existing predictions")
    parser.add_argument(
        "--no-service",
        action="store_true",
        help="do not start uvicorn; use an already running service",
    )
    return parser.parse_args()


def detect_data_root(repo_root: Path) -> Path:
    target = repo_root / "data" / "bird"
    candidates = []
    if (target / "dev.json").is_file():
        candidates.append(target)
    candidates.extend(path.parent for path in target.rglob("dev.json"))
    for candidate in sorted(set(candidates), key=lambda path: len(path.parts)):
        if (candidate / "dev_databases").exists():
            return candidate.resolve()
    raise SystemExit(
        "BIRD Dev data not found. Run: python scripts/prepare_bird_dev.py"
    )


def wait_health(base_url: str, process: subprocess.Popen[str] | None, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/health"
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Python service exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Python service did not become healthy: {last_error}")


def report_score(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("| total |"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 4:
                return {
                    "total": int(cells[1]),
                    "correct": int(cells[2]),
                    "ex": float(cells[3]),
                }
    raise RuntimeError(f"cannot parse total score from {path}")


def run_mode(
    repo_root: Path,
    mode: str,
    data_root: Path,
    base_url: str,
    concurrency: int,
    timeout: float,
    max_rounds: int,
    resume: bool,
) -> Path:
    output = repo_root / "scripts" / "report" / f"pred.python.{mode}.jsonl"
    report = output.with_suffix(".report.md")
    if not resume:
        for path in (output, report):
            if path.exists():
                path.unlink()

    command = [
        sys.executable,
        str(repo_root / "scripts" / "bird_eval_runner.py"),
        "--mode",
        mode,
        "--base-url",
        base_url,
        "--data-root",
        str(data_root),
        "--max-questions",
        "120",
        "--concurrency",
        str(concurrency),
        "--timeout",
        str(timeout),
        "--output",
        str(output),
    ]
    if max_rounds > 0:
        command.extend(["--max-rounds", str(max_rounds)])

    print("[repro] " + " ".join(command))
    completed = subprocess.run(command, cwd=repo_root)
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"{mode} runner failed with code {completed.returncode}")
    if not report.is_file():
        raise RuntimeError(f"missing report: {report}")
    return report


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    backend = repo_root / "python_backend"
    data_root = Path(args.data_root).resolve() if args.data_root else detect_data_root(repo_root)

    process: subprocess.Popen[str] | None = None
    if not args.no_service:
        env = os.environ.copy()
        env["BIRD_EVAL_ENABLED"] = "true"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            [
                "uv",
                "run",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8889",
            ],
            cwd=backend,
            env=env,
            text=True,
            creationflags=creationflags,
        )

    try:
        wait_health(args.base_url, process)
        results = {}
        reports = {}
        for mode in ("baseline", "agent"):
            report = run_mode(
                repo_root,
                mode,
                data_root,
                args.base_url,
                args.concurrency,
                args.timeout,
                args.max_rounds,
                args.resume,
            )
            score = report_score(report)
            results[mode] = score
            reports[mode] = str(report)
            expected = EXPECTED[mode]
            exact = score == expected
            print(
                f"[repro] {mode}: {score['correct']}/{score['total']} "
                f"EX={score['ex']:.2f} | historical={expected['ex']:.2f} "
                f"| exact={'YES' if exact else 'NO'}"
            )

        summary = {
            "data_root": str(data_root),
            "results": results,
            "historical_resume_targets": EXPECTED,
            "exact_reproduction": all(results[mode] == EXPECTED[mode] for mode in EXPECTED),
            "reports": reports,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["exact_reproduction"] else 3
    finally:
        if process is not None:
            stop_process(process)


if __name__ == "__main__":
    sys.exit(main())
