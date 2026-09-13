from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent_graph import AgentGraphFactory
from app.config import get_settings
from app.tool_registry import ToolRegistry


async def main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "charts": {
                "transport": "stdio",
                "command": "npx.cmd",
                "args": ["-y", "mcp-echarts@0.7.1", "-t", "stdio"],
            }
        },
        handle_tool_errors=False,
    )
    mcp_tools = await client.get_tools()

    settings = get_settings()
    factory = AgentGraphFactory(settings)
    registry = ToolRegistry.from_tools([], mcp_tools=mcp_tools)
    graph = factory.build(
        registry,
        (
            "你是 MCP deferred-tool 联调 Agent。"
            "需要图表时必须先调用 tool_search 搜索并加载 MCP 图表工具，"
            "然后调用真实 MCP 工具。"
            "本次测试优先使用 generate_echarts，outputType 使用 option，"
            "避免把 PNG base64 再传给模型。"
        ),
        checkpointer=False,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "请为 A=1、B=3、C=2 生成一个柱状图配置。"
                        "必须通过 MCP ECharts 工具完成，不要自己直接写最终配置。"
                    )
                )
            ],
            "tool_rounds": 0,
            "loaded_tools": [],
        }
    )

    sequence: list[str] = []
    tool_messages: list[ToolMessage] = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            sequence.append(call["name"])
        if isinstance(message, ToolMessage):
            tool_messages.append(message)

    print("LOADED_TOOLS=" + repr(result.get("loaded_tools")))
    print("TOOL_SEQUENCE=" + repr(sequence))
    print("TOOL_ROUNDS=" + str(result.get("tool_rounds")))

    for index, message in enumerate(tool_messages, 1):
        content = str(message.content)
        print(f"TOOL_{index}_NAME={message.name}")
        print(f"TOOL_{index}_CONTENT_PREFIX={content[:1200]}")
        print(f"TOOL_{index}_CONTENT_LENGTH={len(content)}")

    final = result["messages"][-1]
    print("FINAL_CONTENT=" + repr(getattr(final, "content", "")))

    assert sequence, "no tool calls"
    assert sequence[0] == "tool_search", sequence
    mcp_calls = [name for name in sequence[1:] if name != "tool_search"]
    assert mcp_calls, sequence
    assert any(name.startswith("generate_") for name in mcp_calls), mcp_calls
    assert any(message.name in mcp_calls for message in tool_messages)

    print("DEFERRED_MCP_AGENT_OK")


if __name__ == "__main__":
    asyncio.run(main())
