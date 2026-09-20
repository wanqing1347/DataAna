from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from .agent_graph import AgentGraphFactory
from .checkpointing import DurableCheckpointManager
from .config import Settings
from .db import Database
from .events import TimelineRecorder, content_text, sse
from .memory import ConversationMemory
from .models import User
from .prompts import build_system_prompt
from .runtime import AgentRunTrace, ToolBudgetExceeded
from .schema_catalog import SchemaCatalog
from .security import DataScopeRewriter, SensitiveFilter, SqlSafetyGuard
from .session_store import SessionStore
from .tool_registry import ToolRegistry
from .tools import build_tools
from .user_context import UserContextService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActiveRun:
    run_id: str
    stop_event: asyncio.Event


class AgentService:
    # "running" can be stale after a process crash. In-memory self.running still
    # prevents a live run from being resumed concurrently.
    RESUMABLE_STATUSES = {"running", "paused", "failed", "interrupted"}

    def __init__(
        self,
        settings: Settings,
        db: Database,
        catalog: SchemaCatalog,
        graph_factory: AgentGraphFactory,
        scope_service: UserContextService,
        session_store: SessionStore,
        guard: SqlSafetyGuard,
        rewriter: DataScopeRewriter,
        sensitive_filter: SensitiveFilter,
        checkpoint_manager: DurableCheckpointManager,
    ):
        self.settings = settings
        self.db = db
        self.catalog = catalog
        self.graph_factory = graph_factory
        self.scope_service = scope_service
        self.session_store = session_store
        self.guard = guard
        self.rewriter = rewriter
        self.sensitive_filter = sensitive_filter
        self.checkpoints = checkpoint_manager
        self.memory = ConversationMemory(
            session_store,
            max_turns=settings.memory_max_turns,
            max_chars=settings.memory_max_chars,
        )
        self.running: dict[tuple[int, str], ActiveRun] = {}
        self.mcp_tools = []
        self.mcp_client = None
        self.mcp_status: dict[str, Any] = {
            "enabled": bool(settings.chart_mcp_enabled),
            "connected": False,
            "url": settings.chart_mcp_url,
            "tools": [],
            "error": None,
        }

    async def startup(self) -> None:
        await self.checkpoints.startup()
        if not self.settings.chart_mcp_enabled:
            logger.info("chart MCP disabled")
            return
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self.mcp_client = MultiServerMCPClient(
                {
                    "charts": {
                        "transport": "http",
                        "url": self.settings.chart_mcp_url,
                    }
                }
            )
            self.mcp_tools = await self.mcp_client.get_tools()
            tool_names = [tool.name for tool in self.mcp_tools]
            self.mcp_status.update(
                {"connected": True, "tools": tool_names, "error": None}
            )
            logger.info("chart MCP connected: %s tools=%s", self.settings.chart_mcp_url, tool_names)
        except Exception as exc:
            self.mcp_tools = []
            self.mcp_status.update(
                {"connected": False, "tools": [], "error": f"{type(exc).__name__}: {exc}"}
            )
            logger.exception("chart MCP startup failed: %s", self.settings.chart_mcp_url)

    def mcp_health(self) -> dict[str, Any]:
        return dict(self.mcp_status)

    async def shutdown(self) -> None:
        await self.checkpoints.shutdown()

    def stop(self, conversation_id: str, user_id: int) -> str | None:
        active = self.running.get((user_id, conversation_id))
        if active is None:
            return None
        active.stop_event.set()
        return active.run_id

    async def has_checkpoint(self, user_id: int, run_id: str) -> bool:
        config = self._graph_config(user_id, run_id)
        try:
            checkpoint = await self.checkpoints.checkpointer.aget(config)
        except Exception:
            return False
        return checkpoint is not None

    async def delete_session(self, conversation_id: str, user_id: int) -> bool:
        if (user_id, conversation_id) in self.running:
            return False
        run_ids = self.session_store.list_run_ids(user_id, conversation_id)
        deleted = self.session_store.delete(user_id, conversation_id)
        if not deleted:
            return False
        for run_id in run_ids:
            try:
                await self.checkpoints.checkpointer.adelete_thread(
                    self.checkpoints.thread_id(user_id, run_id)
                )
            except Exception:
                # Business data deletion already succeeded; orphan checkpoint cleanup
                # can be handled by retention tooling if storage is temporarily unavailable.
                pass
        return True

    async def stream(self, query: str, conversation_id: str, user: User) -> AsyncIterator[str]:
        run_key = (user.id, conversation_id)
        if run_key in self.running:
            yield self._already_running_event(conversation_id)
            return

        history = self.memory.load(user.id, conversation_id)
        history_chars = sum(len(content_text(message.content)) for message in history)
        prompt = build_system_prompt(self.scope_service.build_prompt_context(user))
        graph = self._build_graph(conversation_id, user, prompt)

        trace = AgentRunTrace(
            conversation_id,
            user.id,
            self.settings.deepseek_model,
            prompt_chars=len(prompt) + len(query) + history_chars,
        )
        initial_state = {
            "messages": [*history, HumanMessage(content=query)],
            "tool_rounds": 0,
            "loaded_tools": [],
        }

        async for chunk in self._stream_graph(
            graph=graph,
            graph_input=initial_state,
            query=query,
            conversation_id=conversation_id,
            user=user,
            trace=trace,
            resumed=False,
        ):
            yield chunk

    async def resume(
        self,
        conversation_id: str,
        run_id: str,
        user: User,
    ) -> AsyncIterator[str]:
        run_key = (user.id, conversation_id)
        if run_key in self.running:
            yield self._already_running_event(conversation_id)
            return

        saved = self.session_store.get_run(user.id, conversation_id, run_id)
        if saved is None:
            yield sse(
                {
                    "type": "Error",
                    "message": "未找到可恢复的 Agent run",
                    "conversationId": conversation_id,
                    "runId": run_id,
                }
            )
            return
        if saved.get("status") not in self.RESUMABLE_STATUSES:
            yield sse(
                {
                    "type": "Error",
                    "message": f"当前 run 状态 {saved.get('status')} 不允许 resume",
                    "conversationId": conversation_id,
                    "runId": run_id,
                }
            )
            return

        prompt = build_system_prompt(self.scope_service.build_prompt_context(user))
        graph = self._build_graph(conversation_id, user, prompt)
        config = self._graph_config(user.id, run_id)
        try:
            snapshot = await graph.aget_state(config)
        except Exception as exc:
            yield sse(
                {
                    "type": "Error",
                    "message": "读取持久化 checkpoint 失败",
                    "detail": str(exc),
                    "runId": run_id,
                }
            )
            return

        if not snapshot.values:
            yield sse(
                {
                    "type": "Error",
                    "message": "该 run 没有可用 checkpoint",
                    "runId": run_id,
                }
            )
            return
        if not snapshot.next:
            yield sse(
                {
                    "type": "Error",
                    "message": "该 run 已没有待执行节点，不能继续 resume",
                    "runId": run_id,
                }
            )
            return

        query = self._latest_human_text(snapshot.values.get("messages") or [])
        trace = AgentRunTrace(
            conversation_id,
            user.id,
            self.settings.deepseek_model,
            prompt_chars=len(prompt) + len(query),
            run_id=run_id,
            resume_count=int(saved.get("resumeCount") or 0) + 1,
        )

        async for chunk in self._stream_graph(
            graph=graph,
            graph_input=None,
            query=query,
            conversation_id=conversation_id,
            user=user,
            trace=trace,
            resumed=True,
        ):
            yield chunk

    def _build_graph(self, conversation_id: str, user: User, prompt: str):
        scope_ctx = self.scope_service.resolve_scope(user)
        local_tools = build_tools(
            self.db,
            self.catalog,
            self.guard,
            self.rewriter,
            self.sensitive_filter,
            scope_ctx,
            conversation_id=conversation_id,
            session_store=self.session_store,
            model=self.graph_factory.model,
            tool_retry_max_attempts=self.settings.tool_retry_max_attempts,
            tool_retry_base_delay_ms=self.settings.tool_retry_base_delay_ms,
            sql_planner_max_attempts=self.settings.sql_planner_max_attempts,
            structured_output_method=self.settings.deepseek_structured_output_method,
        )
        registry = ToolRegistry.from_tools(local_tools, mcp_tools=self.mcp_tools)
        return self.graph_factory.build(
            registry,
            prompt,
            checkpointer=self.checkpoints.checkpointer,
        )

    async def _stream_graph(
        self,
        *,
        graph,
        graph_input: dict[str, Any] | None,
        query: str,
        conversation_id: str,
        user: User,
        trace: AgentRunTrace,
        resumed: bool,
    ) -> AsyncIterator[str]:
        run_key = (user.id, conversation_id)
        stop_event = asyncio.Event()
        self.running[run_key] = ActiveRun(trace.run_id, stop_event)
        recorder = TimelineRecorder()
        config = self._graph_config(user.id, trace.run_id)

        # Persist the control-plane run record before graph execution. If the
        # process crashes, the durable checkpoint can still be discovered and
        # authorized on the next process via this stale "running" record.
        try:
            self.session_store.save_run(trace.snapshot())
        except Exception:
            self.running.pop(run_key, None)
            yield sse(
                {
                    "type": "Error",
                    "message": "无法持久化 Agent run 元数据，已拒绝启动以避免不可恢复执行",
                    "runId": trace.run_id,
                    "resumable": False,
                }
            )
            return

        yield sse(
            {
                "type": "AgentResume" if resumed else "AgentStart",
                "conversationId": conversation_id,
                "runId": trace.run_id,
                "resumed": resumed,
            }
        )

        event_stream = graph.astream_events(
            graph_input,
            config=config,
            version="v2",
        )
        try:
            async for event in event_stream:
                if stop_event.is_set():
                    trace.finish("paused")
                    yield sse(
                        {
                            "type": "Paused",
                            "conversationId": conversation_id,
                            "runId": trace.run_id,
                            "resumable": True,
                        }
                    )
                    break

                event_type = event.get("event")
                name = event.get("name") or ""
                event_run_id = str(event.get("run_id") or "")
                data = event.get("data") or {}
                metadata = event.get("metadata") or {}
                node_name = metadata.get("langgraph_node")
                is_main_model = not node_name or node_name == "model"

                if event_type == "on_chat_model_start" and is_main_model:
                    status = "正在分析问题与工具结果…"
                    recorder.thinking(status)
                    yield sse({"type": "Thinking", "content": status})

                elif event_type == "on_chat_model_stream" and is_main_model:
                    chunk = data.get("chunk")
                    if chunk is None:
                        continue
                    text_part = content_text(getattr(chunk, "content", ""))
                    if text_part:
                        trace.model_chunk(text_part)
                        recorder.text(text_part)
                        yield sse({"type": "Text", "content": text_part})

                elif event_type == "on_tool_start":
                    arguments = data.get("input") or {}
                    trace.tool_start(event_run_id, name)
                    call_id = recorder.tool_start(event_run_id, name, arguments)
                    yield sse(
                        {
                            "type": "ToolStart",
                            "toolName": name,
                            "toolCallId": call_id,
                            "arguments": json.dumps(arguments, ensure_ascii=False, default=str),
                        }
                    )

                elif event_type == "on_tool_end":
                    trace.tool_end(event_run_id, name)
                    call_id, result_text = recorder.tool_end(
                        event_run_id, name, data.get("output")
                    )
                    yield sse(
                        {
                            "type": "ToolEnd",
                            "toolName": name,
                            "toolCallId": call_id,
                            "result": result_text,
                        }
                    )
            else:
                trace.finish("completed")
                yield sse(
                    {
                        "type": "Complete",
                        "conversationId": conversation_id,
                        "runId": trace.run_id,
                        "durationMs": trace.duration_ms,
                        "toolCallCount": len(trace.tools),
                        "resumed": resumed,
                    }
                )
        except asyncio.CancelledError:
            trace.finish("interrupted")
            raise
        except ToolBudgetExceeded as exc:
            trace.finish("budget_exceeded", str(exc))
            yield sse(
                {
                    "type": "Error",
                    "message": "Agent 工具调用次数达到安全上限",
                    "detail": str(exc),
                    "runId": trace.run_id,
                    "resumable": False,
                }
            )
        except Exception as exc:
            trace.finish("failed", str(exc))
            yield sse(
                {
                    "type": "Error",
                    "message": "Agent 执行失败，可使用同一 runId 尝试 resume",
                    "detail": str(exc),
                    "runId": trace.run_id,
                    "resumable": True,
                }
            )
        finally:
            aclose = getattr(event_stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
            if trace.status == "running":
                trace.finish("interrupted")
            self.running.pop(run_key, None)

            if trace.status == "completed":
                try:
                    self.session_store.save_turn(
                        conversation_id,
                        user.id,
                        query,
                        recorder.answer,
                        recorder.items,
                    )
                except Exception:
                    pass
            try:
                self.session_store.save_run(trace.snapshot())
            except Exception:
                pass

    def _graph_config(self, user_id: int, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": self.checkpoints.thread_id(user_id, run_id),
            },
            "recursion_limit": self.settings.agent_recursion_limit,
        }

    @staticmethod
    def _latest_human_text(messages) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return content_text(message.content)
        return ""

    @staticmethod
    def _already_running_event(conversation_id: str) -> str:
        return sse(
            {
                "type": "Error",
                "message": "该会话已有 Agent 任务正在执行",
                "conversationId": conversation_id,
            }
        )
