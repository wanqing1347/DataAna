from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from .db import Database

logger = logging.getLogger(__name__)

MAX_PLAN_ROWS = 64

# MySQL error code for "Query execution was interrupted, max_execution_time exceeded".
_MYSQL_MAX_EXECUTION_TIME_EXCEEDED = 3024


@dataclass(slots=True)
class PrecheckResult:
    allowed: bool
    reason: str

    @classmethod
    def allow(cls) -> "PrecheckResult":
        return cls(True, "OK")

    @classmethod
    def block(cls, reason: str) -> "PrecheckResult":
        return cls(False, reason)


@dataclass(slots=True)
class ExplainRow:
    select_type: str | None = None
    table: str | None = None
    access_type: str | None = None
    key: str | None = None
    rows: int = 0
    extra: str | None = None


class ExplainPrecheck:
    """SQL 执行计划预检查（移植自 Java ExplainPrecheckService）。

    只做保守拦截：小表全表扫描、无索引但数据量小的查询、合理聚合查询都放行；
    只拦截明显高风险的大表明细扫描或 JOIN 结果扩张。

    非超时类失败一律 fail-open 放行，避免预检查本身把正常查询挡掉。
    """

    def __init__(
        self,
        db: Database,
        *,
        enabled: bool = True,
        timeout_seconds: int = 5,
        max_estimated_rows: int = 100_000,
    ):
        self.db = db
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.max_estimated_rows = max_estimated_rows

    def check(self, sql: str) -> PrecheckResult:
        if not self.enabled or not sql or not sql.strip():
            return PrecheckResult.allow()
        try:
            rows = self._explain(sql)
        except OperationalError as exc:
            if _is_timeout(exc):
                reason = "EXPLAIN 预检查超时，说明数据库生成执行计划也较慢，当前 SQL 可能过重"
                logger.warning("%s | sql=%s", reason, _truncate(sql, 300))
                return PrecheckResult.block(reason)
            logger.warning("EXPLAIN 预检查失败，按 fail-open 放行：%s", exc)
            return PrecheckResult.allow()
        except Exception as exc:  # noqa: BLE001 - fail-open, mirroring Java
            logger.warning("EXPLAIN 预检查失败，按 fail-open 放行：%s", exc)
            return PrecheckResult.allow()
        return self.analyze(sql, rows)

    def _explain(self, sql: str) -> list[ExplainRow]:
        rows: list[ExplainRow] = []
        with self.db.connect() as conn:
            self._set_statement_timeout(conn, self.timeout_seconds * 1000)
            try:
                result = conn.execute(text("EXPLAIN " + sql))
                for raw in result.mappings().fetchmany(MAX_PLAN_ROWS):
                    cols = {str(key).lower(): value for key, value in raw.items()}
                    rows.append(
                        ExplainRow(
                            select_type=_as_str(cols.get("select_type")),
                            table=_as_str(cols.get("table")),
                            access_type=_as_str(cols.get("type")),
                            key=_as_str(cols.get("key")),
                            rows=_as_int(cols.get("rows")),
                            extra=_as_str(cols.get("extra")),
                        )
                    )
            finally:
                self._set_statement_timeout(conn, 0)
        return rows

    @staticmethod
    def _set_statement_timeout(conn, milliseconds: int) -> None:
        # MySQL 5.7+ only; best-effort so non-MySQL backends keep working.
        try:
            conn.execute(text(f"SET SESSION max_execution_time = {int(milliseconds)}"))
        except Exception:  # noqa: BLE001
            pass

    def analyze(self, sql: str, rows: list[ExplainRow]) -> PrecheckResult:
        if not rows:
            return PrecheckResult.allow()
        if not _looks_like_detail_query(sql):
            return PrecheckResult.allow()

        table_rows = [row for row in rows if row.table and row.table.strip()]
        if not table_rows:
            return PrecheckResult.allow()

        for row in table_rows:
            if _is_risky_access(row.access_type) and row.rows >= self.max_estimated_rows:
                return PrecheckResult.block(
                    f"疑似大表明细扫描，表 {row.table} 使用 {_safe(row.access_type)} 访问，"
                    f"预计扫描 {row.rows} 行"
                )

        if len(table_rows) >= 2:
            threshold = max(self.max_estimated_rows * 10, self.max_estimated_rows)
            if _estimated_join_rows(table_rows, threshold) >= threshold and _has_join_risk_signal(
                table_rows
            ):
                return PrecheckResult.block(
                    f"疑似 JOIN 结果扩张，预计组合行数超过 {threshold}，"
                    "且存在无索引访问或 Join Buffer 风险"
                )

        return PrecheckResult.allow()


def _looks_like_detail_query(sql: str) -> bool:
    normalized = " ".join(sql.lower().split())
    aggregate_markers = (" group by ", " having ", "count(", "sum(", "avg(", "min(", "max(")
    return not any(marker in normalized for marker in aggregate_markers)


def _has_join_risk_signal(rows: list[ExplainRow]) -> bool:
    missing_key_count = 0
    for index, row in enumerate(rows):
        extra = (row.extra or "").lower()
        if "using join buffer" in extra:
            return True
        if index > 0 and _is_risky_access(row.access_type):
            return True
        if _is_blank_key(row.key):
            missing_key_count += 1
    return missing_key_count >= 2


def _estimated_join_rows(rows: list[ExplainRow], threshold: int) -> int:
    product = 1
    for row in rows:
        value = max(1, row.rows)
        if product > threshold // value:
            return threshold
        product *= value
    return product


def _is_risky_access(access_type: str | None) -> bool:
    return (access_type or "").lower() in {"all", "index"}


def _is_blank_key(key: str | None) -> bool:
    return not key or key.strip().upper() == "NULL"


def _is_timeout(exc: OperationalError) -> bool:
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", None) or ()
    if args and args[0] == _MYSQL_MAX_EXECUTION_TIME_EXCEEDED:
        return True
    message = f"{exc} {orig}".lower()
    return "3024" in message or "max_execution_time" in message

def _as_str(value) -> str | None:
    return None if value is None else str(value)


def _as_int(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe(value: str | None) -> str:
    return value if value and value.strip() else "UNKNOWN"


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "..."
