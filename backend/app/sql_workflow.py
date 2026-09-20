from __future__ import annotations

import asyncio
import json
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .schema_catalog import SchemaCatalog
from .security import SqlSafetyGuard
from .sql_execution import SqlExecutionService


class SqlDraft(BaseModel):
    plan: str = Field(description="简洁说明查询粒度、表、JOIN、过滤和聚合策略")
    sql: str = Field(description="单条 MySQL SELECT/WITH，只能使用给定 schema")


class SqlCritique(BaseModel):
    approved: bool = Field(description="SQL 是否既安全又能回答用户问题")
    feedback: str = Field(description="若不通过，指出需要修正的具体问题；通过时简述检查结论")


class SqlWorkflowState(TypedDict, total=False):
    question: str
    tool_call_id: str
    attempt: int
    plan: str
    sql: str
    safe_sql: str
    critique: str
    approved: bool
    result: str
    error: str


class SqlPlanningWorkflow:
    """Planner -> critic -> executor Text-to-SQL subgraph."""

    def __init__(
        self,
        model: BaseChatModel,
        catalog: SchemaCatalog,
        guard: SqlSafetyGuard,
        executor: SqlExecutionService,
        *,
        max_attempts: int = 3,
        structured_output_method: str = "function_calling",
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        self.catalog = catalog
        self.guard = guard
        self.executor = executor
        self.max_attempts = max_attempts
        self._planner = model.with_structured_output(
            SqlDraft, method=structured_output_method
        )
        self._critic = model.with_structured_output(
            SqlCritique, method=structured_output_method
        )
        self._schema_context = self._build_schema_context()
        self.graph = self._build_graph()

    def _build_schema_context(self) -> str:
        names = sorted(
            name
            for name in self.catalog.tables
            if name.lower() in self.catalog.allowed
        )
        return self.catalog.describe_tables(names)

    def _build_graph(self):
        builder = StateGraph(SqlWorkflowState)
        builder.add_node("planner", self._plan)
        builder.add_node("critic", self._critique)
        builder.add_node("executor", self._execute)
        builder.add_node("failed", self._failed)
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "critic")
        builder.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"planner": "planner", "executor": "executor", "failed": "failed"},
        )
        builder.add_edge("executor", END)
        builder.add_edge("failed", END)
        # This subgraph is a pure planning/validation workflow except for its executor.
        # The executor itself is persistently idempotent, so parent-run resume is safe
        # even if the tool node has to re-enter this subgraph after a process crash.
        return builder.compile(checkpointer=False)

    async def _plan(self, state: SqlWorkflowState) -> dict:
        attempt = int(state.get("attempt", 0)) + 1
        feedback = state.get("critique") or "无"
        prompt = (
            "你是 DataAna 的 SQL planner。根据真实 schema 生成最小、可验证的 MySQL 查询。"
            "禁止猜字段，禁止写操作，默认只返回回答问题所需列。\n\n"
            f"Schema:\n{self._schema_context}\n\n"
            f"用户问题：{state['question']}\n"
            f"上一轮 critic 反馈：{feedback}\n"
            f"当前尝试：{attempt}/{self.max_attempts}"
        )
        draft = await self._planner.ainvoke(
            [
                SystemMessage(content="只做 Text-to-SQL 规划，不回答最终业务结论。"),
                HumanMessage(content=prompt),
            ]
        )
        return {
            "attempt": attempt,
            "plan": draft.plan,
            "sql": draft.sql,
            "approved": False,
        }

    async def _critique(self, state: SqlWorkflowState) -> dict:
        checked = self.guard.validate(state["sql"])
        if not checked.valid or not checked.safe_sql:
            return {
                "approved": False,
                "safe_sql": "",
                "critique": f"确定性 SQL guard 拒绝：{checked.reason}",
            }

        prompt = (
            "你是 DataAna 的 SQL critic。检查 SQL 是否严格回答问题、字段/聚合/过滤/JOIN "
            "是否合理。安全性已有 AST guard，但你仍需关注语义正确性。\n\n"
            f"Schema:\n{self._schema_context}\n\n"
            f"用户问题：{state['question']}\n"
            f"Planner 计划：{state.get('plan', '')}\n"
            f"候选 SQL：{checked.safe_sql}"
        )
        critique = await self._critic.ainvoke(
            [
                SystemMessage(content="不要执行 SQL，只做语义 critic。"),
                HumanMessage(content=prompt),
            ]
        )
        return {
            "approved": bool(critique.approved),
            "safe_sql": checked.safe_sql,
            "critique": critique.feedback,
        }

    def _route_after_critic(self, state: SqlWorkflowState) -> str:
        if state.get("approved"):
            return "executor"
        if int(state.get("attempt", 0)) < self.max_attempts:
            return "planner"
        return "failed"

    async def _execute(self, state: SqlWorkflowState) -> dict:
        sql = state.get("safe_sql") or state["sql"]
        result = await asyncio.to_thread(
            self.executor.execute,
            sql,
            tool_name="queryData",
            tool_call_id=state["tool_call_id"],
            idempotency_arguments={"question": state["question"]},
        )
        return {"result": result}

    async def _failed(self, state: SqlWorkflowState) -> dict:
        return {
            "error": (
                f"SQL planner 在 {state.get('attempt', 0)} 次尝试后仍未通过 critic："
                f"{state.get('critique', '未知原因')}"
            )
        }

    async def run(self, question: str, tool_call_id: str) -> str:
        final = await self.graph.ainvoke(
            {
                "question": question,
                "tool_call_id": tool_call_id,
                "attempt": 0,
            }
        )
        if final.get("error"):
            return json.dumps(
                {
                    "ok": False,
                    "stage": "sql_critic",
                    "error": final["error"],
                    "plan": final.get("plan"),
                    "sql": final.get("sql"),
                    "critique": final.get("critique"),
                },
                ensure_ascii=False,
            )

        execution = final.get("result") or "{}"
        try:
            execution_payload = json.loads(execution)
        except json.JSONDecodeError:
            execution_payload = execution
        return json.dumps(
            {
                "ok": bool(
                    isinstance(execution_payload, dict)
                    and execution_payload.get("ok")
                ),
                "stage": "completed",
                "attempts": final.get("attempt"),
                "plan": final.get("plan"),
                "sql": final.get("safe_sql") or final.get("sql"),
                "critique": final.get("critique"),
                "execution": execution_payload,
            },
            ensure_ascii=False,
        )
