from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from .db import Database


class SessionStore:
    def __init__(self, db: Database):
        self.db = db

    def initialize(self) -> None:
        turn_ddl = """
        CREATE TABLE IF NOT EXISTS py_agent_turn (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            conversation_id VARCHAR(128) NOT NULL,
            user_id BIGINT NOT NULL,
            question TEXT NOT NULL,
            answer MEDIUMTEXT NULL,
            timeline MEDIUMTEXT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_py_turn_user_time (user_id, created_at),
            INDEX idx_py_turn_conv (conversation_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        run_ddl = """
        CREATE TABLE IF NOT EXISTS py_agent_run (
            run_id VARCHAR(64) NOT NULL PRIMARY KEY,
            conversation_id VARCHAR(128) NOT NULL,
            user_id BIGINT NOT NULL,
            status VARCHAR(32) NOT NULL,
            model VARCHAR(128) NOT NULL,
            duration_ms BIGINT NOT NULL,
            tool_call_count INT NOT NULL,
            metrics MEDIUMTEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_py_run_user_conv_time (user_id, conversation_id, created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        tool_result_ddl = """
        CREATE TABLE IF NOT EXISTS py_agent_tool_result (
            idempotency_key VARCHAR(64) NOT NULL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            conversation_id VARCHAR(128) NOT NULL,
            tool_name VARCHAR(128) NOT NULL,
            tool_call_id VARCHAR(255) NOT NULL,
            args_hash VARCHAR(64) NOT NULL,
            status VARCHAR(32) NOT NULL,
            result MEDIUMTEXT NULL,
            attempts INT NOT NULL,
            last_error MEDIUMTEXT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_py_tool_user_conv (user_id, conversation_id),
            INDEX idx_py_tool_call (tool_call_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        with self.db.begin() as conn:
            conn.execute(text(turn_ddl))
            conn.execute(text(run_ddl))
            conn.execute(text(tool_result_ddl))

    def save_turn(
        self,
        conversation_id: str,
        user_id: int,
        question: str,
        answer: str,
        timeline: list[dict[str, Any]],
    ) -> None:
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO py_agent_turn "
                    "(conversation_id, user_id, question, answer, timeline) "
                    "VALUES (:cid, :uid, :q, :a, :t)"
                ),
                {
                    "cid": conversation_id,
                    "uid": user_id,
                    "q": question,
                    "a": answer,
                    "t": json.dumps(timeline, ensure_ascii=False, default=str),
                },
            )

    def recent_turns(self, user_id: int, conversation_id: str, limit: int) -> list[dict]:
        if limit <= 0:
            return []
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT question, answer FROM py_agent_turn "
                    "WHERE user_id = :uid AND conversation_id = :cid "
                    "ORDER BY created_at DESC, id DESC LIMIT :limit"
                ),
                {"uid": user_id, "cid": conversation_id, "limit": limit},
            ).mappings().all()
        return [dict(row) for row in reversed(rows)]

    def save_run(self, metrics: dict[str, Any]) -> None:
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO py_agent_run "
                    "(run_id, conversation_id, user_id, status, model, duration_ms, "
                    "tool_call_count, metrics) "
                    "VALUES (:run_id, :cid, :uid, :status, :model, :duration_ms, "
                    ":tool_call_count, :metrics) "
                    "ON DUPLICATE KEY UPDATE "
                    "status = VALUES(status), model = VALUES(model), "
                    "duration_ms = VALUES(duration_ms), "
                    "tool_call_count = VALUES(tool_call_count), metrics = VALUES(metrics)"
                ),
                {
                    "run_id": metrics["runId"],
                    "cid": metrics["conversationId"],
                    "uid": metrics["userId"],
                    "status": metrics["status"],
                    "model": metrics["model"],
                    "duration_ms": metrics["durationMs"],
                    "tool_call_count": metrics["toolCallCount"],
                    "metrics": json.dumps(metrics, ensure_ascii=False, default=str),
                },
            )

    def get_run(self, user_id: int, conversation_id: str, run_id: str) -> dict | None:
        with self.db.connect() as conn:
            raw = conn.execute(
                text(
                    "SELECT metrics FROM py_agent_run "
                    "WHERE user_id = :uid AND conversation_id = :cid AND run_id = :run_id"
                ),
                {"uid": user_id, "cid": conversation_id, "run_id": run_id},
            ).scalar_one_or_none()
        return json.loads(raw) if raw else None

    def list_runs(self, user_id: int, conversation_id: str, limit: int = 20) -> list[dict]:
        safe_limit = min(max(limit, 1), 100)
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT metrics FROM py_agent_run "
                    "WHERE user_id = :uid AND conversation_id = :cid "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"uid": user_id, "cid": conversation_id, "limit": safe_limit},
            ).scalars().all()
        return [json.loads(row) for row in rows]

    def list_run_ids(self, user_id: int, conversation_id: str) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT run_id FROM py_agent_run "
                    "WHERE user_id = :uid AND conversation_id = :cid"
                ),
                {"uid": user_id, "cid": conversation_id},
            ).scalars().all()
        return [str(row) for row in rows]

    def get_tool_result(self, idempotency_key: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT args_hash, status, result, attempts, last_error "
                    "FROM py_agent_tool_result WHERE idempotency_key = :key"
                ),
                {"key": idempotency_key},
            ).mappings().one_or_none()
        if row is None:
            return None
        return {
            "argsHash": row["args_hash"],
            "status": row["status"],
            "result": row["result"],
            "attempts": row["attempts"],
            "lastError": row["last_error"],
        }

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
    ) -> None:
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO py_agent_tool_result "
                    "(idempotency_key, user_id, conversation_id, tool_name, tool_call_id, "
                    "args_hash, status, result, attempts, last_error) "
                    "VALUES (:key, :uid, :cid, :tool_name, :tool_call_id, :args_hash, "
                    ":status, :result, :attempts, :last_error) "
                    "ON DUPLICATE KEY UPDATE "
                    "args_hash = VALUES(args_hash), status = VALUES(status), "
                    "result = VALUES(result), attempts = VALUES(attempts), "
                    "last_error = VALUES(last_error)"
                ),
                {
                    "key": idempotency_key,
                    "uid": user_id,
                    "cid": conversation_id,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "args_hash": args_hash,
                    "status": status,
                    "result": result,
                    "attempts": attempts,
                    "last_error": last_error,
                },
            )

    def list_conversations(self, user_id: int, page: int, size: int) -> dict:
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, conversation_id, question, created_at "
                    "FROM py_agent_turn WHERE user_id = :uid ORDER BY created_at DESC, id DESC"
                ),
                {"uid": user_id},
            ).mappings().all()

        grouped: dict[str, list] = {}
        for row in rows:
            grouped.setdefault(row["conversation_id"], []).append(row)

        items = []
        for cid, turns in grouped.items():
            latest = turns[0]
            first = turns[-1]
            title = first["question"] or ""
            if len(title) > 40:
                title = title[:40] + "..."
            items.append(
                {
                    "conversationId": cid,
                    "title": title,
                    "messageCount": len(turns),
                    "createdAt": first["created_at"],
                    "updatedAt": latest["created_at"],
                }
            )

        total = len(items)
        start = max(page, 0) * max(size, 1)
        end = start + max(size, 1)
        return {
            "conversations": items[start:end],
            "total": total,
            "page": max(page, 0),
            "size": max(size, 1),
            "hasMore": end < total,
        }

    def detail(self, user_id: int, conversation_id: str) -> dict | None:
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, question, answer, timeline, created_at "
                    "FROM py_agent_turn WHERE user_id = :uid AND conversation_id = :cid "
                    "ORDER BY created_at ASC, id ASC"
                ),
                {"uid": user_id, "cid": conversation_id},
            ).mappings().all()
        if not rows:
            return None
        return {
            "conversationId": conversation_id,
            "messages": [
                {
                    "id": int(r["id"]),
                    "question": r["question"],
                    "answer": r["answer"] or "",
                    "timeline": r["timeline"] or "[]",
                    "createdAt": r["created_at"],
                    "attachments": [],
                }
                for r in rows
            ],
        }

    def delete(self, user_id: int, conversation_id: str) -> bool:
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM py_agent_tool_result "
                    "WHERE user_id = :uid AND conversation_id = :cid"
                ),
                {"uid": user_id, "cid": conversation_id},
            )
            conn.execute(
                text(
                    "DELETE FROM py_agent_run WHERE user_id = :uid AND conversation_id = :cid"
                ),
                {"uid": user_id, "cid": conversation_id},
            )
            result = conn.execute(
                text(
                    "DELETE FROM py_agent_turn WHERE user_id = :uid AND conversation_id = :cid"
                ),
                {"uid": user_id, "cid": conversation_id},
            )
        return bool(result.rowcount)
