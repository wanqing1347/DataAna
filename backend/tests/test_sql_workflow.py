import json

import pytest

from app.security import SqlSafetyGuard
from app.sql_workflow import SqlCritique, SqlDraft, SqlPlanningWorkflow


class FakeRunnable:
    def __init__(self, outputs):
        self.outputs = outputs

    async def ainvoke(self, messages):
        return self.outputs.pop(0)


class FakeModel:
    def __init__(self):
        self.drafts = [
            SqlDraft(plan="bad", sql="DELETE FROM payment"),
            SqlDraft(plan="safe", sql="SELECT payment_id FROM payment LIMIT 5"),
        ]
        self.critiques = [
            SqlCritique(approved=True, feedback="字段、粒度和过滤满足问题"),
        ]

    def with_structured_output(self, schema, **kwargs):
        if schema is SqlDraft:
            return FakeRunnable(self.drafts)
        if schema is SqlCritique:
            return FakeRunnable(self.critiques)
        raise AssertionError(schema)


class FakeCatalog:
    tables = {"payment": {}}
    allowed = {"payment"}

    def describe_tables(self, names):
        return "## payment\n- payment_id | BIGINT"


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, **kwargs):
        self.calls.append((sql, kwargs))
        return json.dumps({"ok": True, "rows": [{"payment_id": 1}]})


@pytest.mark.asyncio
async def test_sql_subgraph_replans_after_deterministic_critic_rejection():
    executor = FakeExecutor()
    workflow = SqlPlanningWorkflow(
        FakeModel(),
        FakeCatalog(),
        SqlSafetyGuard(["payment"], max_rows=200, max_joins=3),
        executor,
        max_attempts=3,
    )

    raw = await workflow.run("查询支付记录", "tool-call-1")
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["attempts"] == 2
    assert "SELECT payment_id FROM payment" in payload["sql"]
    assert len(executor.calls) == 1
    assert executor.calls[0][1]["tool_call_id"] == "tool-call-1"
