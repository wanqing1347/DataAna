from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .config import Settings
from .runtime import ToolExecutionBudget
from .tool_registry import ToolRegistry, extract_loaded_tool_names


class AgentState(MessagesState):
    tool_rounds: int
    loaded_tools: list[str]


class AgentGraphFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.deepseek_temperature,
            timeout=300,
            max_retries=2,
        )

    def build(self, registry: ToolRegistry, system_prompt: str, *, checkpointer):
        budget = ToolExecutionBudget(self.settings.agent_max_tool_rounds)

        async def call_model(state: AgentState):
            visible_tools = registry.bindable_tools(state.get("loaded_tools", []))
            model_with_tools = self.model.bind_tools(visible_tools)
            response = await model_with_tools.ainvoke(
                [SystemMessage(content=system_prompt), *state["messages"]]
            )
            return {"messages": [response]}

        async def call_tools(state: AgentState):
            completed_rounds = int(state.get("tool_rounds", 0))
            budget.ensure_available(completed_rounds)
            loaded_before = list(state.get("loaded_tools", []))
            tool_node = ToolNode(registry.bindable_tools(loaded_before))
            result = await tool_node.ainvoke(state)
            discovered = extract_loaded_tool_names(result["messages"])
            loaded_after = list(dict.fromkeys([*loaded_before, *discovered]))
            return {
                "messages": result["messages"],
                "tool_rounds": completed_rounds + 1,
                "loaded_tools": loaded_after,
            }

        def should_continue(state: AgentState):
            last = state["messages"][-1]
            return "tools" if getattr(last, "tool_calls", None) else END

        builder = StateGraph(AgentState)
        builder.add_node("model", call_model)
        builder.add_node("tools", call_tools)
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "model")
        return builder.compile(checkpointer=checkpointer)
