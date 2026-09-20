from __future__ import annotations

import json
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .models import DataScope, DataScopeContext


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    reason: str
    safe_sql: str | None = None

    def as_json(self) -> str:
        return json.dumps(
            {"valid": self.valid, "reason": self.reason, "safeSql": self.safe_sql},
            ensure_ascii=False,
        )


class SqlSafetyGuard:
    DANGEROUS_FUNCTIONS = {
        "load_file",
        "sleep",
        "benchmark",
        "sys_eval",
        "sys_exec",
    }

    def __init__(self, allowed_tables: list[str], max_rows: int = 200, max_joins: int = 3):
        self.allowed_tables = {t.lower() for t in allowed_tables}
        self.max_rows = max_rows
        self.max_joins = max_joins

    def validate(self, sql: str) -> ValidationResult:
        if not sql or not sql.strip():
            return ValidationResult(False, "SQL 不能为空")
        if ";" in sql.strip().rstrip(";"):
            return ValidationResult(False, "只允许执行单条 SQL")

        try:
            statements = sqlglot.parse(sql, read="mysql")
        except Exception as exc:
            return ValidationResult(False, f"SQL 解析失败：{exc}")

        if len(statements) != 1:
            return ValidationResult(False, "只允许执行单条 SQL")
        tree = statements[0]
        if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return ValidationResult(False, "只允许 SELECT/WITH 只读查询")

        tables = {t.name.lower() for t in tree.find_all(exp.Table)}
        denied = sorted(t for t in tables if t not in self.allowed_tables)
        if denied:
            return ValidationResult(
                False,
                f"表 {', '.join(denied)} 不在 DataAna 数据分析白名单中，请先调用 listTables。",
            )

        joins = list(tree.find_all(exp.Join))
        if len(joins) > self.max_joins:
            return ValidationResult(False, f"JOIN 数量超过上限 {self.max_joins}")
        for join in joins:
            kind = str(join.args.get("kind") or "").upper()
            if kind == "CROSS":
                return ValidationResult(False, "禁止 CROSS JOIN")
            if join.args.get("on") is None and not join.args.get("using"):
                return ValidationResult(False, "禁止无条件 JOIN，必须提供 ON/USING")

        for fn in tree.find_all(exp.Func):
            name = (getattr(fn, "name", "") or fn.sql_name() or "").lower()
            if name in self.DANGEROUS_FUNCTIONS:
                return ValidationResult(False, f"禁止调用危险函数 {name}")

        safe_tree = tree.copy()
        current_limit = self._literal_limit(safe_tree)
        if current_limit is None or current_limit > self.max_rows:
            safe_tree = safe_tree.limit(self.max_rows)
        return ValidationResult(True, "OK", safe_tree.sql(dialect="mysql"))

    @staticmethod
    def _literal_limit(tree: exp.Expression) -> int | None:
        limit = tree.args.get("limit")
        if not limit:
            return None
        value = limit.args.get("expression")
        if isinstance(value, exp.Literal) and value.is_int:
            return int(value.this)
        return None


class DataScopeRewriter:
    MANAGED_TABLES = {"payment", "rental"}

    def rewrite(self, sql: str, ctx: DataScopeContext) -> str:
        if ctx.scope == DataScope.ALL:
            return sql
        tree = sqlglot.parse_one(sql, read="mysql")

        for select in list(tree.find_all(exp.Select)):
            where_conditions: list[exp.Expression] = []

            from_clause = select.args.get("from_")
            if from_clause is not None and isinstance(from_clause.this, exp.Table):
                cond = self._condition(from_clause.this, ctx)
                if cond is not None:
                    where_conditions.append(cond)

            for join in select.args.get("joins") or []:
                table = join.this
                if not isinstance(table, exp.Table):
                    continue
                cond = self._condition(table, ctx)
                if cond is None:
                    continue
                side = str(join.args.get("side") or "").upper()
                if side == "LEFT" and join.args.get("on") is not None:
                    join.set("on", exp.and_(join.args["on"], cond))
                else:
                    where_conditions.append(cond)

            if where_conditions:
                combined = where_conditions[0]
                for cond in where_conditions[1:]:
                    combined = exp.and_(combined, cond)
                old_where = select.args.get("where")
                if old_where is not None:
                    combined = exp.and_(old_where.this, combined)
                select.set("where", exp.Where(this=combined))

        return tree.sql(dialect="mysql")

    def _condition(self, table: exp.Table, ctx: DataScopeContext) -> exp.Expression | None:
        if table.name.lower() not in self.MANAGED_TABLES:
            return None
        qualifier = table.alias_or_name
        if ctx.scope == DataScope.SELF:
            return exp.column("user_id", table=qualifier).eq(exp.Literal.number(ctx.user_id))
        values = [exp.Literal.number(x) for x in ctx.dept_ids] or [exp.Null()]
        return exp.column("dept_id", table=qualifier).isin(*values)


class SensitiveFilter:
    MASK = "********"

    def __init__(self, enabled: bool, fields: list[str]):
        self.enabled = enabled
        self.columns = {
            item.lower().rsplit(".", 1)[-1]
            for item in fields
            if item and item.strip()
        }

    def mask_rows(self, rows: list[dict]) -> list[dict]:
        if not self.enabled:
            return rows
        for row in rows:
            for key in list(row):
                if key.lower() in self.columns and row[key] is not None:
                    row[key] = self.MASK
        return rows
