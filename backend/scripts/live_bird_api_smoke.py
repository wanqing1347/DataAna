from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def execute(path: Path, sql: str):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchall()


def main() -> None:
    db_path = Path(".data/langgraph_checkpoints.sqlite3").resolve()
    assert db_path.is_file(), db_path

    payload = {
        "questionId": "smoke-1",
        "dbId": "checkpoint_smoke",
        "question": "How many rows are in the checkpoints table? Return only the count.",
        "evidence": "",
        "sqlitePath": str(db_path),
        "maxRounds": 2,
    }

    with TestClient(app) as client:
        baseline = client.post("/bird/eval/baseline", json=payload)
        print("BASELINE_STATUS=" + str(baseline.status_code))
        baseline_json = baseline.json()
        print("BASELINE_JSON=" + repr(baseline_json))
        assert baseline.status_code == 200
        assert baseline_json.get("success") is True, baseline_json
        baseline_sql = str(baseline_json.get("sql") or "")
        baseline_rows = execute(db_path, baseline_sql)
        print("BASELINE_ROWS=" + repr(baseline_rows))
        assert baseline_rows == [(0,)], baseline_rows

        agent = client.post("/bird/eval/question", json=payload)
        print("AGENT_STATUS=" + str(agent.status_code))
        agent_json = agent.json()
        print("AGENT_JSON=" + repr(agent_json))
        assert agent.status_code == 200
        assert agent_json.get("success") is True, agent_json
        agent_sql = str(agent_json.get("sql") or "")
        agent_rows = execute(db_path, agent_sql)
        print("AGENT_ROWS=" + repr(agent_rows))
        assert agent_rows == [(0,)], agent_rows
        assert agent_json.get("approved") is True, agent_json

    print("BIRD_API_SMOKE_OK")


if __name__ == "__main__":
    main()
