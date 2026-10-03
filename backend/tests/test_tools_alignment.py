from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.explain_precheck import ExplainPrecheck, ExplainRow
from app.models import DataScope, DataScopeContext
from app.schema_catalog import SchemaCatalog
from app.security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from app.tools import SafeCalculator, build_tools


# ── calculate ──────────────────────────────────────────────────────────────


def test_calculator_supports_variables_and_power_operator():
    calculator = SafeCalculator({"current": 3298, "previous": 3105})
    result = calculator.eval("(current - previous) / previous * 100")
    assert round(result, 4) == round((3298 - 3105) / 3105 * 100, 4)
    assert calculator.eval("2 ^ 10") == 1024
    assert calculator.eval("2 ** 10") == 1024


def test_calculator_supports_functions_and_round():
    calculator = SafeCalculator()
    assert calculator.eval("round(10 / 3, 2)") == 3.33
    assert calculator.eval("sqrt(16)") == 4.0
    assert calculator.eval("max(1, 5, 3)") == 5
    assert round(calculator.eval("log(exp(1))"), 6) == 1.0


def test_calculator_rejects_unknown_variable_and_function():
    calculator = SafeCalculator()
    with pytest.raises(ValueError, match="未定义变量"):
        calculator.eval("missing + 1")
    with pytest.raises(ValueError, match="不支持的函数"):
        calculator.eval("evil(1)")


def test_calculator_rejects_non_math_syntax():
    calculator = SafeCalculator()
    with pytest.raises(ValueError):
        calculator.eval("__import__('os').system('ls')")
    with pytest.raises(ValueError):
        calculator.eval("(1).__class__")


def _build_local_tools():
    settings = Settings()
    catalog = SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)
    tools = build_tools(
        object(),
        catalog,
        SqlSafetyGuard(settings.allowed_tables, settings.max_rows, settings.max_joins),
        DataScopeRewriter(),
        SensitiveFilter(settings.sensitive_filter_enabled, settings.sensitive_fields),
        DataScopeContext(user_id=1, scope=DataScope.ALL, dept_ids=[]),
        conversation_id="test",
        session_store=object(),
        tool_retry_max_attempts=settings.tool_retry_max_attempts,
        tool_retry_base_delay_ms=settings.tool_retry_base_delay_ms,
    )
    return {tool.name: tool for tool in tools}


def test_query_data_tool_is_removed():
    tools = _build_local_tools()
    assert list(tools) == [
        "listTables",
        "describeTables",
        "lookupGlossary",
        "validateSql",
        "executeSql",
        "calculate",
    ]
    assert "queryData" not in tools


def test_calculate_tool_accepts_variables_json_and_reports_errors():
    tool = _build_local_tools()["calculate"]
    assert "variablesJson" in tool.args

    payload = json.loads(tool.invoke({"expression": "(a - b) / b * 100", "variablesJson": '{"a": 10, "b": 8}'}))
    assert payload["result"] == 25

    division = json.loads(tool.invoke({"expression": "1 / 0"}))
    assert "除数为 0" in division["error"]

    bad_var = json.loads(tool.invoke({"expression": "a", "variablesJson": '{"c": 1}'}))
    assert "未定义变量" in bad_var["error"]


# ── lookupGlossary ─────────────────────────────────────────────────────────


def test_catalog_aligned_with_java_schema():
    settings = Settings()
    catalog = SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)
    assert len(settings.allowed_tables) == 20
    assert len(catalog.tables) == 20
    assert len(catalog.glossary) == 21
    # 这些术语在收窄到 4 张表时缺失，现在应与 Java 一致可用
    for term in ["近6个月", "近1年", "本月", "上月", "今年", "热门影片", "高消费客户", "库存可用", "销售业绩", "部门收入", "逾期未还", "客户生命周期价值"]:
        assert "未找到术语" not in catalog.lookup_glossary(term), term
    # 此前不在白名单的表现在可被 describeTables 正常描述
    assert "库存拷贝" in catalog.describe_tables(["inventory"])
    assert "data_scope" in catalog.describe_tables(["sys_role"])


def _catalog() -> SchemaCatalog:
    settings = Settings()
    return SchemaCatalog(settings.resolved_schema_path(), settings.allowed_tables)


def test_lookup_glossary_matches_term_and_synonym_exactly():
    catalog = _catalog()
    by_term = catalog.lookup_glossary("活跃客户")
    assert "【术语】活跃客户" in by_term
    assert "【SQL 片段（可直接复用）】" in by_term

    by_synonym = catalog.lookup_glossary("有效客户")
    assert "【术语】活跃客户" in by_synonym


def test_lookup_glossary_miss_lists_registered_terms():
    catalog = _catalog()
    miss = catalog.lookup_glossary("不存在的口径")
    assert "未找到术语" in miss
    assert "活跃客户" in miss


def test_lookup_glossary_does_not_fuzzy_match_inside_description():
    catalog = _catalog()
    # "租赁" only appears inside the 租金 description, not as a term/synonym on its own.
    assert "未找到术语" in catalog.lookup_glossary("租赁")


def test_glossary_tool_description_embeds_term_list():
    tool = _build_local_tools()["lookupGlossary"]
    assert "活跃客户" in tool.description
    assert "同义词" in tool.description


# ── describeTables Examples ────────────────────────────────────────────────


def test_describe_tables_renders_examples():
    text = _catalog().describe_tables(["film"])
    assert "rating" in text
    assert "Examples: [G, PG, PG-13, R, NC-17]" in text


# ── EXPLAIN precheck ───────────────────────────────────────────────────────


def test_explain_precheck_blocks_large_table_detail_scan():
    precheck = ExplainPrecheck(None, max_estimated_rows=100_000)
    rows = [ExplainRow(table="payment", access_type="ALL", key=None, rows=500_000)]
    result = precheck.analyze("SELECT payment_id FROM payment", rows)
    assert result.allowed is False
    assert "大表明细扫描" in result.reason


def test_explain_precheck_allows_aggregate_and_small_scan():
    precheck = ExplainPrecheck(None, max_estimated_rows=100_000)
    rows = [ExplainRow(table="payment", access_type="ALL", key=None, rows=500_000)]
    assert precheck.analyze("SELECT COUNT(*) FROM payment", rows).allowed is True
    small = [ExplainRow(table="film", access_type="ALL", key=None, rows=1000)]
    assert precheck.analyze("SELECT film_id FROM film", small).allowed is True


def test_explain_precheck_blocks_join_expansion():
    precheck = ExplainPrecheck(None, max_estimated_rows=100)
    rows = [
        ExplainRow(table="payment", access_type="ALL", key=None, rows=40),
        ExplainRow(table="rental", access_type="ALL", key=None, rows=40),
        ExplainRow(table="customer", access_type="ALL", key=None, rows=40),
    ]
    sql = (
        "SELECT p.payment_id FROM payment p "
        "JOIN rental r ON p.rental_id = r.rental_id "
        "JOIN customer c ON p.customer_id = c.customer_id"
    )
    result = precheck.analyze(sql, rows)
    assert result.allowed is False
    assert "JOIN" in result.reason


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def fetchmany(self, _limit):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, statement):
        rendered = str(statement)
        if rendered.upper().startswith("EXPLAIN"):
            return _FakeResult(self._rows)
        return _FakeResult([])


class _FakeDb:
    def __init__(self, rows=None, exc=None):
        self._rows = rows or []
        self._exc = exc

    @contextmanager
    def connect(self):
        if self._exc is not None:
            raise self._exc
        yield _FakeConn(self._rows)


def test_explain_precheck_fails_open_on_unexpected_error():
    precheck = ExplainPrecheck(_FakeDb(exc=RuntimeError("boom")))
    assert precheck.check("SELECT 1 FROM payment").allowed is True


def test_explain_precheck_blocks_on_timeout():
    orig = Exception(3024, "Query execution was interrupted, max_execution_time exceeded")
    timeout = OperationalError("EXPLAIN SELECT 1 FROM payment", {}, orig)
    precheck = ExplainPrecheck(_FakeDb(exc=timeout))
    result = precheck.check("SELECT 1 FROM payment")
    assert result.allowed is False
    assert "超时" in result.reason


def test_explain_precheck_reads_plan_rows_from_db():
    rows = [
        {
            "select_type": "SIMPLE",
            "table": "payment",
            "type": "ALL",
            "key": None,
            "rows": 500_000,
            "Extra": "Using where",
        }
    ]
    precheck = ExplainPrecheck(_FakeDb(rows=rows), max_estimated_rows=100_000)
    result = precheck.check("SELECT payment_id FROM payment")
    assert result.allowed is False
