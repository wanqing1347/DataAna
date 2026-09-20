import pytest

from app.runtime import AgentRunTrace, ToolBudgetExceeded, ToolExecutionBudget


def test_tool_budget_fails_closed():
    budget = ToolExecutionBudget(max_rounds=2)
    budget.ensure_available(0)
    budget.ensure_available(1)
    with pytest.raises(ToolBudgetExceeded):
        budget.ensure_available(2)


def test_run_trace_records_structured_metrics_without_reasoning():
    ticks = iter([0.0, 1.0, 1.4, 2.0])
    trace = AgentRunTrace(
        "conv-1",
        7,
        "deepseek-chat",
        120,
        run_id="run-test",
        clock=lambda: next(ticks),
    )
    trace.model_chunk("abc")
    trace.tool_start("tool-run-1", "executeSql")
    trace.tool_end("tool-run-1", "executeSql")
    trace.finish()

    data = trace.snapshot()
    assert data["runId"] == "run-test"
    assert data["status"] == "completed"
    assert data["durationMs"] == 2000
    assert data["outputChars"] == 3
    assert data["toolCallCount"] == 1
    assert data["tools"][0]["name"] == "executeSql"
    assert "reasoning" not in data
