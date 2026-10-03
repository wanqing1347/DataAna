from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from .db import Database
from .explain_precheck import ExplainPrecheck
from .models import DataScopeContext
from .security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from .tool_runtime import IdempotentToolRunner


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, Decimal)):
        return str(value)
    return str(value)


def is_retryable_db_error(exc: Exception) -> bool:
    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    return isinstance(exc, DBAPIError) and bool(exc.connection_invalidated)


class SqlExecutionService:
    """Deterministic SQL safety, scope enforcement, retry and result masking."""

    def __init__(
        self,
        db: Database,
        guard: SqlSafetyGuard,
        rewriter: DataScopeRewriter,
        sensitive_filter: SensitiveFilter,
        scope_ctx: DataScopeContext,
        tool_runner: IdempotentToolRunner,
        explain_precheck: ExplainPrecheck | None = None,
    ):
        self.db = db
        self.guard = guard
        self.rewriter = rewriter
        self.sensitive_filter = sensitive_filter
        self.scope_ctx = scope_ctx
        self.tool_runner = tool_runner
        self.explain_precheck = explain_precheck

    def execute(
        self,
        sql: str,
        *,
        tool_name: str,
        tool_call_id: str,
        idempotency_arguments: dict[str, Any],
    ) -> str:
        checked = self.guard.validate(sql)
        if not checked.valid or not checked.safe_sql:
            return checked.as_json()

        rewritten = self.rewriter.rewrite(checked.safe_sql, self.scope_ctx)

        def query() -> str:
            # 执行计划预检查只在实际要跑查询时做（幂等缓存命中时会跳过）。
            if self.explain_precheck is not None:
                precheck = self.explain_precheck.check(rewritten)
                if not precheck.allowed:
                    return json.dumps(
                        {
                            "ok": False,
                            "stage": "explain_precheck",
                            "error": f"已拦截（EXPLAIN 预检查）。原因：{precheck.reason}",
                            "suggestion": (
                                "请重写 SQL：统计问题优先用 GROUP BY 聚合；明细问题增加选择性 WHERE，"
                                "并按主表主键或时间字段稳定分页；一对多关系请先对子表聚合后 JOIN，"
                                "或先查主键列表再按主键二次钻取。"
                            ),
                        },
                        ensure_ascii=False,
                    )
            with self.db.connect() as conn:
                result = conn.execute(text(rewritten))
                rows = [dict(r) for r in result.mappings().fetchmany(self.guard.max_rows)]
            self.sensitive_filter.mask_rows(rows)
            return json.dumps(
                {
                    "ok": True,
                    "rowCount": len(rows),
                    "columns": list(rows[0].keys()) if rows else [],
                    "rows": rows,
                    "sql": rewritten,
                },
                ensure_ascii=False,
                default=_json_default,
            )

        try:
            run = self.tool_runner.run(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=idempotency_arguments,
                operation=query,
                retry_on=is_retryable_db_error,
            )
            payload = json.loads(run.value)
            if isinstance(payload, dict):
                payload["runtime"] = {
                    "cacheHit": run.cache_hit,
                    "attempts": run.attempts,
                    "idempotencyKey": run.idempotency_key[:16],
                }
                return json.dumps(payload, ensure_ascii=False, default=_json_default)
            return run.value
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"SQL 执行失败：{exc}",
                    "suggestion": "如果是瞬时连接错误可 resume 当前 run；如果是 SQL 错误请重新规划字段、JOIN 或时间条件。",
                },
                ensure_ascii=False,
            )
