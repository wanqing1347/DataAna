from __future__ import annotations

import json
import uuid
from typing import Any


def sse(event: dict[str, Any]) -> str:
    return "data:" + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return ""


class TimelineRecorder:
    def __init__(self):
        self.items: list[dict[str, Any]] = []
        self.answer_parts: list[str] = []
        self.tool_runs: dict[str, str] = {}

    def thinking(self, content: str):
        if content:
            self.items.append({"type": "thinking", "content": content})

    def text(self, content: str):
        if content:
            self.answer_parts.append(content)
            self.items.append({"type": "text", "content": content})

    def tool_start(self, run_id: str, name: str, arguments: Any) -> str:
        call_id = "py_" + uuid.uuid4().hex[:12]
        self.tool_runs[run_id] = call_id
        self.items.append(
            {
                "type": "tool",
                "toolName": name,
                "toolCallId": call_id,
                "arguments": json.dumps(arguments, ensure_ascii=False, default=str),
                "status": "running",
                "result": "",
            }
        )
        return call_id

    def tool_end(self, run_id: str, name: str, result: Any) -> tuple[str, str]:
        call_id = self.tool_runs.get(run_id, "py_" + uuid.uuid4().hex[:12])
        result_text = getattr(result, "content", result)
        if not isinstance(result_text, str):
            result_text = json.dumps(result_text, ensure_ascii=False, default=str)
        for item in reversed(self.items):
            if item.get("type") == "tool" and item.get("toolCallId") == call_id:
                item["status"] = "completed"
                item["result"] = result_text
                break
        return call_id, result_text

    @property
    def answer(self) -> str:
        return "".join(self.answer_parts)
