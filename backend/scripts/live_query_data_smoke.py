from __future__ import annotations

import asyncio
import json
import os
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent_graph import AgentGraphFactory
from app.config import get_settings
from app.db import Database
from app.models import DataScope, DataScopeContext
from app.schema_catalog import SchemaCatalog
from app.security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from app.session_store import SessionStore
from app.tool_registry import ToolRegistry
from app.tools import build_tools


def resolve_database_url(settings) -> str:
    return os.getenv("LIVE_DATABASE_URL") or settings.database_url


async def main() -> None:
    settings = get_settings()
    database_url = resolve_database_url(settings)

    db = Database(database_url)
    session_store = SessionStore(db)
    session_store.initialize()

    catalog = SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)
    guard = SqlSafetyGuard(settings.allowed_tables, settings.max_rows, settings.max_joins)
    rewriter = DataScopeRewriter()
    sensitive_filter = SensitiveFilter(
        settings.sensitive_filter_enabled,
        settings.sensitive_fields,
    )
    scope_ctx = DataScopeContext(user_id=1, scope=DataScope.ALL, dept_ids=[])
    factory = AgentGraphFactory(settings)

    tools = build_tools(
        db,
        catalog,
        guard,
        rewriter,
        sensitive_filter,
        scope_ctx,
        conversation_id="live-query-data-smoke",
        session_store=session_store,
        model=factory.model,
        tool_retry_max_attempts=settings.tool_retry_max_attempts,
        tool_retry_base_delay_ms=settings.tool_retry_base_delay_ms,
        sql_planner_max_attempts=settings.sql_planner_max_attempts,
        structured_output_method=settings.deepseek_structured_output_method,
    )
    registry = ToolRegistry.from_tools(tools)
    graph = factory.build(
        registry,
        (
            "你是 DataAna 的真实联调 Agent。涉及数据库统计时必须先调用 tool_search，"
            "然后优先调用 queryData；不得直接猜测数据或自己编造 SQL 执行结果。"
            "拿到工具结果后，用一句话给出最终答案。"
        ),
        checkpointer=False,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "请统计 payment 表的支付记录总数和 amount 总金额。"
                        "必须通过真实数据库工具完成。"
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

    query_payload = None
    for index, message in enumerate(tool_messages, start=1):
        print(f"TOOL_{index}_NAME={message.name}")
        print(f"TOOL_{index}_CONTENT={message.content}")
        if message.name == "queryData":
            try:
                query_payload = json.loads(message.content)
            except json.JSONDecodeError:
                query_payload = None

    print("FINAL_CONTENT=" + repr(result["messages"][-1].content))

    assert sequence, "no tool calls"
    assert sequence[0] == "tool_search", sequence
    assert "queryData" in sequence, sequence
    assert sequence.index("tool_search") < sequence.index("queryData"), sequence
    assert isinstance(query_payload, dict), query_payload
    assert query_payload.get("ok") is True, query_payload

    execution = query_payload.get("execution")
    assert isinstance(execution, dict) and execution.get("ok") is True, execution
    rows = execution.get("rows") or []
    assert rows, execution

    row = rows[0]
    print("QUERY_ROW=" + repr(row))
    print("QUERY_SQL=" + str(query_payload.get("sql")))
    print("QUERY_ATTEMPTS=" + str(query_payload.get("attempts")))
    print("QUERYDATA_AGENT_OK")

    db.dispose()


if __name__ == "__main__":
    asyncio.run(main())
