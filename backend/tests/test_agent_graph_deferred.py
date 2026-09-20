from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from app.agent_graph import AgentGraphFactory
from app.tool_registry import ToolRegistry


def make_query_tool():
    def query_data(question: str) -> str:
        return f"result:{question}"

    return StructuredTool.from_function(
        query_data,
        name="queryData",
        description="根据问题生成并执行 SQL 查询数据",
    )


class RecordingBoundModel:
    def __init__(self, owner, visible_names):
        self.owner = owner
        self.visible_names = visible_names

    async def ainvoke(self, messages):
        turn = len(self.owner.bind_history)
        if turn == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_search",
                        "args": {"query": "查询业务数据 SQL"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="done")


class RecordingModel:
    def __init__(self):
        self.bind_history = []

    def bind_tools(self, tools):
        names = [tool.name for tool in tools]
        self.bind_history.append(names)
        return RecordingBoundModel(self, names)


@pytest.mark.asyncio
async def test_langgraph_binds_deferred_tool_only_after_tool_search():
    factory = object.__new__(AgentGraphFactory)
    factory.settings = SimpleNamespace(agent_max_tool_rounds=3)
    factory.model = RecordingModel()
    registry = ToolRegistry.from_tools([make_query_tool()])

    graph = factory.build(registry, "test", checkpointer=False)
    result = await graph.ainvoke(
        {"messages": [], "tool_rounds": 0, "loaded_tools": []}
    )

    assert factory.model.bind_history[0] == ["tool_search"]
    assert factory.model.bind_history[1] == ["tool_search", "queryData"]
    assert result["loaded_tools"] == ["queryData"]
    assert result["messages"][-1].content == "done"
