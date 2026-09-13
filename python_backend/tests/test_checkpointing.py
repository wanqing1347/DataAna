from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from app.checkpointing import DurableCheckpointManager


class CounterState(TypedDict):
    value: int


@pytest.mark.asyncio
async def test_durable_checkpoint_resumes_after_manager_restart(tmp_path):
    calls = {"prepare": 0, "unstable": 0}
    should_fail = {"value": True}

    async def prepare(state: CounterState):
        calls["prepare"] += 1
        return {"value": state["value"] + 1}

    async def unstable(state: CounterState):
        calls["unstable"] += 1
        if should_fail["value"]:
            raise ConnectionError("temporary failure")
        return {"value": state["value"] + 1}

    def build(checkpointer):
        builder = StateGraph(CounterState)
        builder.add_node("prepare", prepare)
        builder.add_node("unstable", unstable)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "unstable")
        builder.add_edge("unstable", END)
        return builder.compile(checkpointer=checkpointer)

    path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "resume-test"}}

    manager = DurableCheckpointManager(path)
    await manager.startup()
    graph = build(manager.checkpointer)
    with pytest.raises(ConnectionError):
        await graph.ainvoke({"value": 0}, config)
    await manager.shutdown()

    assert calls["prepare"] == 1
    assert calls["unstable"] == 1

    should_fail["value"] = False
    manager2 = DurableCheckpointManager(path)
    await manager2.startup()
    graph2 = build(manager2.checkpointer)
    result = await graph2.ainvoke(None, config)
    await manager2.shutdown()

    assert result["value"] == 2
    assert calls["prepare"] == 1
    assert calls["unstable"] == 2
