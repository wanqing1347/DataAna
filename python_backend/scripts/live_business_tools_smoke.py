from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agent_graph import AgentGraphFactory
from app.config import get_settings
from app.schema_catalog import SchemaCatalog
from app.tool_registry import ToolRegistry


def collect(result: dict) -> tuple[list[str], list[ToolMessage]]:
    sequence: list[str] = []
    tool_messages: list[ToolMessage] = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            sequence.append(call["name"])
        if isinstance(message, ToolMessage):
            tool_messages.append(message)
    return sequence, tool_messages


async def run_case(
    factory: AgentGraphFactory,
    registry: ToolRegistry,
    question: str,
    required_tool: str,
) -> None:
    graph = factory.build(
        registry,
        (
            "你是 DataAna 工具链的真实联调 Agent。"
            "数据库/schema/术语相关任务必须先调用 tool_search 搜索能力，"
            "然后调用已加载的业务工具获得事实，禁止直接凭常识回答。"
        ),
        checkpointer=False,
    )
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=question)],
            "tool_rounds": 0,
            "loaded_tools": [],
        }
    )
    sequence, tool_messages = collect(result)

    print("QUESTION=" + question)
    print("LOADED_TOOLS=" + repr(result.get("loaded_tools")))
    print("TOOL_SEQUENCE=" + repr(sequence))
    for idx, message in enumerate(tool_messages, 1):
        print(f"TOOL_{idx}_NAME={message.name}")
        print(f"TOOL_{idx}_CONTENT={message.content}")
    print("FINAL_CONTENT=" + repr(result["messages"][-1].content))

    assert sequence, "model did not call any tools"
    assert sequence[0] == "tool_search", sequence
    assert required_tool in sequence, sequence
    assert sequence.index("tool_search") < sequence.index(required_tool), sequence
    assert any(message.name == required_tool for message in tool_messages)
    print(f"CASE_OK={required_tool}")
    print("---")


async def main() -> None:
    settings = get_settings()
    catalog = SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)
    factory = AgentGraphFactory(settings)

    tools = [
        StructuredTool.from_function(
            catalog.list_tables,
            name="listTables",
            description="列出当前 DataAna 允许查询的业务表及其说明。",
        ),
        StructuredTool.from_function(
            catalog.describe_tables,
            name="describeTables",
            description="查看指定表的真实字段、类型、注释和外键。",
        ),
        StructuredTool.from_function(
            catalog.lookup_glossary,
            name="lookupGlossary",
            description="按关键词查询业务指标/术语口径，避免凭经验猜口径。",
        ),
    ]
    registry = ToolRegistry.from_tools(tools)

    await run_case(
        factory,
        registry,
        "请告诉我当前允许分析哪些业务表。必须使用工具获取真实信息。",
        "listTables",
    )
    await run_case(
        factory,
        registry,
        "请查看 payment 表有哪些字段。必须使用工具读取真实 schema。",
        "describeTables",
    )

    print("BUSINESS_DEFERRED_TOOLS_OK")


if __name__ == "__main__":
    asyncio.run(main())
