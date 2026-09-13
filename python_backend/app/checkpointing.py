from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class DurableCheckpointManager:
    """Owns the lifecycle of the async durable LangGraph checkpointer."""

    def __init__(self, path: Path):
        self.path = path
        self._context: Any | None = None
        self._checkpointer: AsyncSqliteSaver | None = None

    async def startup(self) -> None:
        if self._checkpointer is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._context = AsyncSqliteSaver.from_conn_string(str(self.path))
        self._checkpointer = await self._context.__aenter__()
        await self._checkpointer.setup()

    async def shutdown(self) -> None:
        context = self._context
        self._context = None
        self._checkpointer = None
        if context is not None:
            await context.__aexit__(None, None, None)

    @property
    def checkpointer(self) -> AsyncSqliteSaver:
        if self._checkpointer is None:
            raise RuntimeError("checkpoint manager 尚未启动")
        return self._checkpointer

    @staticmethod
    def thread_id(user_id: int, run_id: str) -> str:
        return f"user:{user_id}:run:{run_id}"
