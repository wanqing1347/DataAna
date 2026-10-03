from __future__ import annotations

import json

import pytest

from app.prompts import build_system_prompt
from app.todo import TODO_TOOL_NAME, TodoItem, build_todo_tool
from app.tool_registry import ToolRegistry


def _todos(*items: tuple[str, str, str]) -> list[dict[str, str]]:
    return [
        {"content": content, "activeForm": active_form, "status": status}
        for content, active_form, status in items
    ]


def test_todo_tool_name_and_schema_match_java():
    tool = build_todo_tool()
    assert tool.name == TODO_TOOL_NAME == "TodoWrite"
    assert "todos" in tool.args

    schema = tool.args_schema.model_json_schema()
    assert schema.get("additionalProperties") is False
    assert schema["required"] == ["todos"]
    assert schema["properties"]["todos"]["items"]["$ref"] == "#/$defs/TodoItem"

    item = schema["$defs"]["TodoItem"]
    assert set(item["required"]) == {"content", "activeForm", "status"}
    assert item.get("additionalProperties") is False
    assert item["properties"]["status"]["enum"] == ["pending", "in_progress", "completed"]


def test_todo_tool_is_always_visible_next_to_tool_search():
    registry = ToolRegistry.from_tools([], always_tools=[build_todo_tool()])
    assert [tool.name for tool in registry.initial_tools()] == ["TodoWrite", "tool_search"]
    assert [tool.name for tool in registry.bindable_tools()] == ["TodoWrite", "tool_search"]


def test_todo_tool_returns_counts_for_valid_list():
    tool = build_todo_tool()
    payload = json.loads(
        tool.invoke(
            {
                "todos": _todos(
                    ("探查表结构", "正在探查表结构", "completed"),
                    ("生成 SQL", "正在生成 SQL", "in_progress"),
                    ("出报告", "正在出报告", "pending"),
                )
            }
        )
    )
    assert payload["ok"] is True
    assert payload["total"] == 3
    assert payload["pending"] == 1
    assert payload["inProgress"] == 1
    assert payload["completed"] == 1
    assert [item["status"] for item in payload["todos"]] == ["completed", "in_progress", "pending"]


def test_todo_tool_rejects_multiple_in_progress():
    tool = build_todo_tool()
    payload = json.loads(
        tool.invoke(
            {
                "todos": _todos(
                    ("A", "正在做 A", "in_progress"),
                    ("B", "正在做 B", "in_progress"),
                )
            }
        )
    )
    assert payload["ok"] is False
    assert "只能有一个 in_progress" in payload["error"]


def test_todo_tool_rejects_blank_fields_and_empty_list():
    tool = build_todo_tool()
    blank = json.loads(tool.invoke({"todos": _todos(("  ", "正在做", "pending"))}))
    assert blank["ok"] is False and "content" in blank["error"]

    blank_form = json.loads(tool.invoke({"todos": _todos(("A", "", "pending"))}))
    assert blank_form["ok"] is False and "activeForm" in blank_form["error"]

    empty = json.loads(tool.invoke({"todos": []}))
    assert empty["ok"] is False


def test_todo_tool_rejects_unknown_status_via_validation():
    tool = build_todo_tool()
    with pytest.raises(Exception):
        tool.invoke({"todos": _todos(("A", "正在做 A", "done"))})


def test_todo_item_forbids_extra_fields():
    with pytest.raises(Exception):
        TodoItem(content="A", activeForm="B", status="pending", extra="nope")


def test_system_prompt_documents_task_management_rules():
    prompt = build_system_prompt("运行上下文")
    assert "TodoWrite" in prompt
    assert "任务管理规则" in prompt
    assert "同一时间只能有一个 in_progress" not in prompt  # rules live in the tool description
