from __future__ import annotations

import ast
import json
import operator
from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import InjectedToolCallId, StructuredTool

from .db import Database
from .models import DataScopeContext
from .schema_catalog import SchemaCatalog
from .security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from .session_store import SessionStore
from .sql_execution import SqlExecutionService
from .sql_workflow import SqlPlanningWorkflow
from .tool_runtime import IdempotentToolRunner


class SafeCalculator:
    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval(self, expression: str) -> float | int:
        node = ast.parse(expression, mode="eval").body
        return self._eval(node)

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPS:
            return self.OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.OPS:
            return self.OPS[type(node.op)](self._eval(node.operand))
        raise ValueError("只允许数字和 + - * / // % ** 运算")


def build_tools(
    db: Database,
    catalog: SchemaCatalog,
    guard: SqlSafetyGuard,
    rewriter: DataScopeRewriter,
    sensitive_filter: SensitiveFilter,
    scope_ctx: DataScopeContext,
    *,
    conversation_id: str,
    session_store: SessionStore,
    model: BaseChatModel,
    tool_retry_max_attempts: int,
    tool_retry_base_delay_ms: int,
    sql_planner_max_attempts: int,
    structured_output_method: str = "function_calling",
):
    calculator = SafeCalculator()
    runner = IdempotentToolRunner(
        session_store,
        user_id=scope_ctx.user_id,
        conversation_id=conversation_id,
        max_attempts=tool_retry_max_attempts,
        base_delay_ms=tool_retry_base_delay_ms,
    )
    sql_executor = SqlExecutionService(
        db,
        guard,
        rewriter,
        sensitive_filter,
        scope_ctx,
        runner,
    )
    sql_workflow = SqlPlanningWorkflow(
        model,
        catalog,
        guard,
        sql_executor,
        max_attempts=sql_planner_max_attempts,
        structured_output_method=structured_output_method,
    )

    def list_tables() -> str:
        """列出当前 DataAna 允许查询的业务表及其说明。"""
        return catalog.list_tables()

    def describe_tables(table_names: list[str]) -> str:
        """查看指定表的真实字段、类型、注释和外键。"""
        return catalog.describe_tables(table_names)

    def lookup_glossary(keyword: str) -> str:
        """按关键词查询业务指标/术语口径，避免凭经验猜口径。"""
        return catalog.lookup_glossary(keyword)

    def validate_sql(sql: str) -> str:
        """低层 SQL 调试工具：只做 AST 白名单、只读性、JOIN 和 LIMIT 校验。"""
        return guard.validate(sql).as_json()

    def execute_sql(
        sql: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """低层只读 SQL 执行工具；带持久化幂等、瞬时错误 retry、安全校验和权限注入。"""
        return sql_executor.execute(
            sql,
            tool_name="executeSql",
            tool_call_id=tool_call_id,
            idempotency_arguments={"sql": sql},
        )

    async def query_data(
        question: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """首选数据查询工具：通过 planner -> critic -> executor 子图生成、审查并执行 SQL。"""
        return await sql_workflow.run(question, tool_call_id)

    def calculate(expression: str) -> str:
        """执行确定性数学计算，适合比例、环比、贡献度等派生指标。"""
        try:
            return json.dumps(
                {"expression": expression, "result": calculator.eval(expression)},
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return [
        StructuredTool.from_function(list_tables, name="listTables"),
        StructuredTool.from_function(describe_tables, name="describeTables"),
        StructuredTool.from_function(lookup_glossary, name="lookupGlossary"),
        StructuredTool.from_function(validate_sql, name="validateSql"),
        StructuredTool.from_function(execute_sql, name="executeSql"),
        StructuredTool.from_function(coroutine=query_data, name="queryData"),
        StructuredTool.from_function(calculate, name="calculate"),
    ]
