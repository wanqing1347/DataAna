from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "bird_eval_runner.py"
SPEC = importlib.util.spec_from_file_location("bird_eval_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_parse_question_ids_accepts_commas_and_spaces():
    assert runner.parse_question_ids("38, 50 58,89") == {"38", "50", "58", "89"}


def test_derive_regression_question_ids_is_python_wrong_minus_java_wrong(tmp_path):
    java = tmp_path / "java.md"
    python = tmp_path / "python.md"
    java.write_text(
        "# report\n### question_id=2\n### question_id=4\n",
        encoding="utf-8",
    )
    python.write_text(
        "# report\n### question_id=2\n### question_id=3\n### question_id=10\n",
        encoding="utf-8",
    )

    assert runner.derive_regression_question_ids(java, python) == ["3", "10"]


def test_frozen_java_correct_python_wrong_regression_manifest():
    manifest = REPO_ROOT / "scripts" / "report" / "bird_java_correct_python_wrong.json"

    ids = runner.load_regression_manifest(manifest)

    assert len(ids) == 20
    assert {"38", "50", "58", "76", "89", "98", "99", "103", "111", "119"} <= set(ids)


def test_transient_generation_error_detection_is_specific():
    assert runner.is_transient_generation_error(
        "APIError: Upstream stream ended before terminal chunk"
    )
    assert runner.is_transient_generation_error("Server disconnected without sending a response")
    assert not runner.is_transient_generation_error("ValueError: invalid BirdAgentOutput structured output")


def test_process_question_retries_transient_stream_failure(monkeypatch, tmp_path):
    db_root = tmp_path / "dbs"
    db_dir = db_root / "demo"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "demo.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE t (value INTEGER)")
        connection.execute("INSERT INTO t(value) VALUES (1)")

    responses = iter(
        [
            {
                "success": False,
                "sql": "",
                "error": "APIError: Upstream stream ended before terminal chunk",
                "durationMs": 10,
            },
            {
                "success": True,
                "sql": "SELECT value FROM t",
                "durationMs": 20,
                "attempts": 1,
            },
        ]
    )
    calls = []

    def fake_call_eval(args, payload):
        calls.append(payload)
        return next(responses)

    monkeypatch.setattr(runner, "call_eval", fake_call_eval)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    args = SimpleNamespace(
        db_root=str(db_root),
        max_rounds=50,
        generation_retry=2,
    )
    output = tmp_path / "pred.jsonl"
    row = {
        "question_id": "16",
        "db_id": "demo",
        "question": "How many?",
        "evidence": "",
    }

    question_id, success, duration_ms = runner.process_question(row, args, output)

    assert question_id == "16"
    assert success is True
    assert duration_ms == 20
    assert len(calls) == 2
    assert calls[0]["maxRounds"] == 50
    record = runner.load_predictions(output)["16"]
    assert record["pred"] == "SELECT value FROM t"
    assert "error" not in record


def test_append_prediction_persists_agent_diagnostics(tmp_path):
    output = tmp_path / "pred.jsonl"
    diagnostics = {
        "approved": True,
        "toolRounds": 3,
        "probeCalls": 1,
        "requestedColumns": ["Phone"],
        "toolHistory": [{"name": "verifySql", "passed": True}],
    }

    runner.append_prediction(
        output,
        "7",
        "california_schools",
        "SELECT Phone FROM schools",
        duration_ms=123,
        diagnostics=diagnostics,
    )

    record = runner.load_predictions(output)["7"]
    assert record["diagnostics"] == diagnostics
