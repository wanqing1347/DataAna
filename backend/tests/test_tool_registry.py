import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from app.tool_registry import ToolRegistry, extract_loaded_tool_names


def make_tool(name: str, description: str):
    def echo(value: str = "") -> str:
        return value

    return StructuredTool.from_function(echo, name=name, description=description)


def registry():
    local_tools = [
        make_tool("listTables", "列出当前允许查询的业务表"),
        make_tool("describeTables", "查看表字段和表结构"),
        make_tool("lookupGlossary", "查询业务指标术语口径"),
        make_tool("queryData", "根据问题生成并执行 SQL 查询数据"),
        make_tool("calculate", "执行比例、环比等数学计算"),
    ]
    mcp_tools = [
        make_tool("generate_echarts", "Generate an ECharts chart and return an image URL"),
    ]
    return ToolRegistry.from_tools(local_tools, mcp_tools=mcp_tools)


def test_initial_context_only_exposes_tool_search():
    current = registry()
    assert [tool.name for tool in current.initial_tools()] == ["tool_search"]
    assert [tool.name for tool in current.bindable_tools()] == ["tool_search"]


def test_tool_search_discovers_and_then_allows_matching_deferred_tools():
    current = registry()
    result = current.search("查询业务数据并生成 SQL")
    names = [item["name"] for item in result]

    assert "queryData" in names
    visible = [tool.name for tool in current.bindable_tools(names)]
    assert visible[0] == "tool_search"
    assert "queryData" in visible
    assert "generate_echarts" not in visible


def test_mcp_chart_tool_is_deferred_until_chart_search():
    current = registry()
    names = [item["name"] for item in current.search("把结果画成 echarts 图表")]
    assert "generate_echarts" in names

    before = [tool.name for tool in current.bindable_tools()]
    after = [tool.name for tool in current.bindable_tools(names)]
    assert "generate_echarts" not in before
    assert "generate_echarts" in after


def test_tool_search_message_promotes_loaded_tools_into_graph_state():
    current = registry()
    payload = current.search_tool.invoke({"query": "指标口径"})
    parsed = json.loads(payload)
    message = ToolMessage(
        content=payload,
        name="tool_search",
        tool_call_id="call-search-1",
    )

    loaded = extract_loaded_tool_names([message])
    assert loaded == parsed["loaded_tools"]
    assert "lookupGlossary" in loaded


def test_initial_schema_context_is_materially_smaller_than_eager_binding():
    current = registry()
    initial = current.context_metrics()
    eager = current.eager_context_metrics()

    assert initial.tool_count == 1
    assert eager.tool_count == 7
    assert initial.schema_chars < eager.schema_chars
    assert initial.approx_tokens < eager.approx_tokens
    assert initial.schema_chars / eager.schema_chars < 0.35
