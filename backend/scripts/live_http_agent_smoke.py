from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import app, auth_service, db


def parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :]
        events.append(json.loads(payload))
    return events


def main() -> None:
    with db.connect() as conn:
        user_id = conn.execute(
            text(
                "SELECT id FROM sys_user "
                "WHERE status = 'ACTIVE' ORDER BY id ASC LIMIT 1"
            )
        ).scalar()
    assert user_id is not None, "no active user available for HTTP smoke test"

    token = auth_service._encode_token(int(user_id))
    headers = {"satoken": token}
    conversation_id = "http-smoke-" + uuid4().hex[:12]

    with TestClient(app) as client:
        me = client.get("/auth/me", headers=headers)
        print("AUTH_ME_STATUS=" + str(me.status_code))
        assert me.status_code == 200

        with client.stream(
            "POST",
            "/agent/stream",
            headers=headers,
            json={
                "query": "请告诉我当前允许分析哪些业务表。必须使用工具获取真实信息。",
                "conversationId": conversation_id,
            },
        ) as response:
            print("STREAM_STATUS=" + str(response.status_code))
            print("STREAM_CONTENT_TYPE=" + str(response.headers.get("content-type")))
            assert response.status_code == 200
            body = "".join(response.iter_text())

        events = parse_sse(body)
        event_types = [event.get("type") for event in events]
        tool_starts = [
            event.get("toolName")
            for event in events
            if event.get("type") == "ToolStart"
        ]
        text_output = "".join(
            str(event.get("content") or "")
            for event in events
            if event.get("type") == "Text"
        )

        print("EVENT_TYPES=" + repr(event_types))
        print("TOOL_STARTS=" + repr(tool_starts))
        print("FINAL_TEXT=" + repr(text_output))
        assert "AgentStart" in event_types
        assert "Complete" in event_types
        assert "Error" not in event_types
        assert tool_starts, "HTTP agent did not emit tool events"
        assert tool_starts[0] == "tool_search", tool_starts
        assert "listTables" in tool_starts, tool_starts
        assert text_output.strip(), "HTTP agent returned no final text"

        detail = client.get(f"/session/{conversation_id}", headers=headers)
        print("SESSION_DETAIL_STATUS=" + str(detail.status_code))
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload.get("code") == 200

        runs = client.get(
            f"/session/{conversation_id}/runs",
            headers=headers,
        )
        print("SESSION_RUNS_STATUS=" + str(runs.status_code))
        assert runs.status_code == 200
        runs_payload = runs.json()
        assert runs_payload.get("code") == 200

        deleted = client.delete(f"/session/{conversation_id}", headers=headers)
        print("SESSION_DELETE_STATUS=" + str(deleted.status_code))
        assert deleted.status_code == 200

    print("HTTP_AGENT_STREAM_OK")


if __name__ == "__main__":
    main()
