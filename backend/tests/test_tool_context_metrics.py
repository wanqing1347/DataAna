from scripts.tool_context_metrics import measure


def test_production_local_tool_context_metric_reduces_initial_schema():
    result = measure()

    assert result["deferredToolNames"] == [
        "listTables",
        "describeTables",
        "lookupGlossary",
        "validateSql",
        "executeSql",
        "queryData",
        "calculate",
    ]
    assert result["initial"]["toolCount"] == 1
    assert result["eager"]["toolCount"] == 8
    assert result["initial"]["schemaChars"] < result["eager"]["schemaChars"]
    assert result["schemaCharReductionPct"] > 80
