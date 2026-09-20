from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4


class ToolBudgetExceeded(RuntimeError):
    pass


class ToolExecutionBudget:
    def __init__(self, max_rounds: int):
        if max_rounds < 1:
            raise ValueError("max_rounds 必须大于等于 1")
        self.max_rounds = max_rounds

    def ensure_available(self, completed_rounds: int) -> None:
        if completed_rounds >= self.max_rounds:
            raise ToolBudgetExceeded(
                f"Agent 工具调用轮次超过上限 {self.max_rounds}，已停止继续执行。"
            )


class AgentRunTrace:
    """Structured runtime telemetry. Never stores model chain-of-thought."""

    def __init__(
        self,
        conversation_id: str,
        user_id: int,
        model: str,
        prompt_chars: int,
        *,
        run_id: str | None = None,
        resume_count: int = 0,
        clock: Callable[[], float] = perf_counter,
    ):
        self.run_id = run_id or ("run_" + uuid4().hex)
        self.resume_count = max(0, int(resume_count))
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.model = model
        self.prompt_chars = prompt_chars
        self.started_at = datetime.now(UTC)
        self._clock = clock
        self._started = clock()
        self._ended: float | None = None
        self.status = "running"
        self.error: str | None = None
        self.model_chunks = 0
        self.output_chars = 0
        self._active_tools: dict[str, tuple[str, float]] = {}
        self.tools: list[dict[str, Any]] = []

    def model_chunk(self, text: str) -> None:
        if not text:
            return
        self.model_chunks += 1
        self.output_chars += len(text)

    def tool_start(self, event_run_id: str, name: str) -> None:
        self._active_tools[event_run_id] = (name, self._clock())

    def tool_end(self, event_run_id: str, name: str) -> None:
        started = self._active_tools.pop(event_run_id, None)
        duration_ms = 0
        if started is not None:
            _, started_at = started
            duration_ms = max(0, round((self._clock() - started_at) * 1000))
        self.tools.append(
            {
                "name": name,
                "status": "completed",
                "durationMs": duration_ms,
            }
        )

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        if self._ended is not None:
            return
        now = self._clock()
        self._ended = now
        self.status = status
        self.error = error
        for _, (name, started_at) in list(self._active_tools.items()):
            self.tools.append(
                {
                    "name": name,
                    "status": "interrupted",
                    "durationMs": max(0, round((now - started_at) * 1000)),
                }
            )
        self._active_tools.clear()

    @property
    def duration_ms(self) -> int:
        ended = self._ended if self._ended is not None else self._clock()
        return max(0, round((ended - self._started) * 1000))

    def snapshot(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "resumeCount": self.resume_count,
            "conversationId": self.conversation_id,
            "userId": self.user_id,
            "model": self.model,
            "status": self.status,
            "startedAt": self.started_at.isoformat(),
            "durationMs": self.duration_ms,
            "promptChars": self.prompt_chars,
            "outputChars": self.output_chars,
            "modelChunks": self.model_chunks,
            "toolCallCount": len(self.tools),
            "tools": list(self.tools),
            "error": self.error,
        }
