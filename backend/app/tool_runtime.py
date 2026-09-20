from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar


T = TypeVar("T")


class IdempotencyConflict(RuntimeError):
    pass


class ToolResultStore(Protocol):
    def get_tool_result(self, idempotency_key: str) -> dict[str, Any] | None: ...

    def save_tool_result(
        self,
        *,
        idempotency_key: str,
        user_id: int,
        conversation_id: str,
        tool_name: str,
        tool_call_id: str,
        args_hash: str,
        status: str,
        result: str | None,
        attempts: int,
        last_error: str | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolRunResult:
    value: str
    cache_hit: bool
    attempts: int
    idempotency_key: str


def canonical_args_hash(arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_idempotency_key(
    *,
    user_id: int,
    conversation_id: str,
    tool_name: str,
    tool_call_id: str,
) -> str:
    raw = f"{user_id}|{conversation_id}|{tool_name}|{tool_call_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class IdempotentToolRunner:
    """Persistent idempotency + bounded retry for one logical tool invocation."""

    def __init__(
        self,
        store: ToolResultStore,
        *,
        user_id: int,
        conversation_id: str,
        max_attempts: int,
        base_delay_ms: int,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if base_delay_ms < 0:
            raise ValueError("base_delay_ms 不能小于 0")
        self.store = store
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.max_attempts = max_attempts
        self.base_delay_ms = base_delay_ms

    def run(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        operation: Callable[[], str],
        retry_on: Callable[[Exception], bool],
    ) -> ToolRunResult:
        key = make_idempotency_key(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
        args_hash = canonical_args_hash(arguments)
        cached = self.store.get_tool_result(key)
        if cached and cached.get("argsHash") != args_hash:
            raise IdempotencyConflict(
                f"幂等键冲突：{tool_name}/{tool_call_id} 被用于不同参数"
            )
        if cached and cached.get("status") == "completed":
            return ToolRunResult(
                value=str(cached.get("result") or ""),
                cache_hit=True,
                attempts=int(cached.get("attempts") or 1),
                idempotency_key=key,
            )

        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                value = operation()
                self.store.save_tool_result(
                    idempotency_key=key,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    args_hash=args_hash,
                    status="completed",
                    result=value,
                    attempts=attempts,
                    last_error=None,
                )
                return ToolRunResult(
                    value=value,
                    cache_hit=False,
                    attempts=attempts,
                    idempotency_key=key,
                )
            except Exception as exc:
                retryable = retry_on(exc)
                self.store.save_tool_result(
                    idempotency_key=key,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    args_hash=args_hash,
                    status="failed",
                    result=None,
                    attempts=attempts,
                    last_error=str(exc),
                )
                if not retryable or attempts >= self.max_attempts:
                    raise
                if self.base_delay_ms:
                    delay = (self.base_delay_ms / 1000.0) * (2 ** (attempts - 1))
                    time.sleep(delay)

        raise AssertionError("unreachable")
