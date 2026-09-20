from __future__ import annotations

import asyncio
import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agent_graph import AgentGraphFactory
from app.config import get_settings
from app.tool_registry import ToolRegistry


def calculate(expression: str) -> str:
    """执行确定性数学计算；用于验证 deferred tool 是否真实加载并执行。"""
    allowed = set("0123456789+-*/(). %")
    if not expression or any(ch not in allowed for ch in expression):
        raise ValueError("unsupported expression")
    result = eval(expression, {"__builtins__": {}}, {})
    return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)


async def main() -> None:
    settings = get_settings()
    factory = AgentGraphFactory(settings)
    tool = StructuredTool.from_function(calculate, name="calculate")
    registry = ToolRegistry.from_tools([tool])

    graph = factory.build(
        registry,
        (
            "你是一个严格使用工具的测试 Agent。"
            "首次需要计算时必须先调用 tool_search 搜索并加载计算工具；"
            "工具加载后必须调用 calculate 完成计算，最后只给出计算结果。"
        ),
        checkpointer=False,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="请计算 (17 * 23) + 9。必须使用工具完成。")
            ],
            "tool_rounds": 0,
            "loaded_tools": [],
        }
    )

    tool_sequence: list[str] = []
    for message in result["messages"]:
        calls = getattr(message, "tool_calls", None) or []
        for call in calls:
            tool_sequence.append(call["name"])

    print("LOADED_TOOLS=" + repr(result.get("loaded_tools")))
    print("TOOL_SEQUENCE=" + repr(tool_sequence))
    print("TOOL_ROUNDS=" + str(result.get("tool_rounds")))

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    for index, message in enumerate(tool_messages, start=1):
        print(f"TOOL_MESSAGE_{index}_NAME={message.name}")
        print(f"TOOL_MESSAGE_{index}_CONTENT={message.content}")

    final = result["messages"][-1]
    print("FINAL_CONTENT=" + repr(getattr(final, "content", "")))

    assert tool_sequence[:2] == ["tool_search", "calculate"], tool_sequence
    assert result.get("loaded_tools") == ["calculate"], result.get("loaded_tools")
    assert any(message.name == "calculate" for message in tool_messages)
    print("DEFERRED_AGENT_OK")


if __name__ == "__main__":
    asyncio.run(main())
