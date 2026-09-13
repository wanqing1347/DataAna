from __future__ import annotations

import json
import re
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import app, auth_service, db


def parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:") :]))
    return events


def main() -> None:
    with db.connect() as conn:
        user_id = conn.execute(
            text(
                "SELECT id FROM sys_user "
                "WHERE status = 'ACTIVE' ORDER BY id ASC LIMIT 1"
            )
        ).scalar()
    assert user_id is not None

    token = auth_service._encode_token(int(user_id))
    headers = {"satoken": token}
    conversation_id = "http-chart-" + uuid4().hex[:12]

    with TestClient(app) as client:
        health = client.get("/health")
        health_json = health.json()
        print("MCP_HEALTH=" + repr(health_json.get("mcp")))
        assert health_json["mcp"]["enabled"] is True
        assert health_json["mcp"]["connected"] is True
        assert health_json["mcp"]["tools"]

        with client.stream(
            "POST",
            "/agent/stream",
            headers=headers,
            json={
                "query": (
                    "请为 A=1、B=3、C=2 生成一个柱状图。"
                    "必须使用 MCP ECharts 工具，并返回图表结果。"
                ),
                "conversationId": conversation_id,
            },
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        events = parse_sse(body)
        tool_starts = [
            event.get("toolName")
            for event in events
            if event.get("type") == "ToolStart"
        ]
        tool_results = [
            str(event.get("result") or "")
            for event in events
            if event.get("type") == "ToolEnd"
        ]
        event_types = [event.get("type") for event in events]

        print("EVENT_TYPES=" + repr(event_types))
        print("TOOL_STARTS=" + repr(tool_starts))
        print("TOOL_RESULTS=" + repr(tool_results))

        assert "Error" not in event_types
        assert "Complete" in event_types
        assert tool_starts
        assert tool_starts[0] == "tool_search", tool_starts
        chart_tools = [
            name
            for name in tool_starts
            if isinstance(name, str)
            and (name.startswith("generate_") or "chart" in name.lower())
        ]
        assert chart_tools, tool_starts

        combined = "\n".join(tool_results)
        urls = re.findall(r"https?://[^\s<>()\"']+", combined)
        print("CHART_URLS=" + repr(urls))
        assert urls, combined
        assert any("/dataana-charts/" in url for url in urls), urls

        deleted = client.delete(f"/session/{conversation_id}", headers=headers)
        assert deleted.status_code == 200

    print("HTTP_AGENT_MCP_MINIO_OK")


if __name__ == "__main__":
    main()
