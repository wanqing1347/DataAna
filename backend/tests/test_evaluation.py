from app.evaluation import ToolPathEvalCase, score_tool_path, score_trace


def test_tool_path_eval_scores_required_and_forbidden_tools():
    case = ToolPathEvalCase(
        case_id="safe-query",
        question="query",
        required_tools=("listTables", "executeSql"),
        forbidden_tools=("dangerousTool",),
    )
    result = score_tool_path(case, ["listTables", "executeSql"])

    assert result["passed"] is True
    assert result["coverage"] == 1.0


def test_tool_path_eval_detects_policy_violation():
    case = ToolPathEvalCase(
        case_id="unsafe",
        question="delete",
        required_tools=(),
        forbidden_tools=("executeSql",),
    )
    result = score_tool_path(case, ["executeSql"])

    assert result["passed"] is False
    assert result["forbiddenToolHits"] == ["executeSql"]


def test_score_trace_reads_runtime_tool_spans():
    case = ToolPathEvalCase(
        case_id="trace",
        question="query",
        required_tools=("listTables", "executeSql"),
    )
    result = score_trace(
        case,
        {
            "runId": "run-1",
            "status": "completed",
            "tools": [{"name": "listTables"}, {"name": "executeSql"}],
        },
    )

    assert result["passed"] is True
    assert result["runId"] == "run-1"
    assert result["runStatus"] == "completed"
