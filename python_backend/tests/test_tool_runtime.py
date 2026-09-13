import pytest

from app.tool_runtime import IdempotencyConflict, IdempotentToolRunner


class FakeStore:
    def __init__(self):
        self.records = {}

    def get_tool_result(self, idempotency_key):
        return self.records.get(idempotency_key)

    def save_tool_result(self, **payload):
        self.records[payload["idempotency_key"]] = {
            "argsHash": payload["args_hash"],
            "status": payload["status"],
            "result": payload["result"],
            "attempts": payload["attempts"],
            "lastError": payload["last_error"],
        }


def runner(store):
    return IdempotentToolRunner(
        store,
        user_id=7,
        conversation_id="conv-1",
        max_attempts=3,
        base_delay_ms=0,
    )


def test_tool_retry_then_persisted_idempotency_cache():
    store = FakeStore()
    calls = {"count": 0}

    def operation():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient")
        return '{"ok": true}'

    first = runner(store).run(
        tool_name="queryData",
        tool_call_id="call-1",
        arguments={"question": "revenue"},
        operation=operation,
        retry_on=lambda exc: isinstance(exc, ConnectionError),
    )
    assert first.cache_hit is False
    assert first.attempts == 3
    assert calls["count"] == 3

    second = runner(store).run(
        tool_name="queryData",
        tool_call_id="call-1",
        arguments={"question": "revenue"},
        operation=lambda: (_ for _ in ()).throw(AssertionError("must not execute")),
        retry_on=lambda exc: False,
    )
    assert second.cache_hit is True
    assert second.value == '{"ok": true}'
    assert calls["count"] == 3


def test_idempotency_key_rejects_same_call_id_with_different_arguments():
    store = FakeStore()
    runner(store).run(
        tool_name="executeSql",
        tool_call_id="call-2",
        arguments={"sql": "SELECT 1"},
        operation=lambda: '{"ok": true}',
        retry_on=lambda exc: False,
    )

    with pytest.raises(IdempotencyConflict):
        runner(store).run(
            tool_name="executeSql",
            tool_call_id="call-2",
            arguments={"sql": "SELECT 2"},
            operation=lambda: '{"ok": true}',
            retry_on=lambda exc: False,
        )
