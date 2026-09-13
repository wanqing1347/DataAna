from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .session_store import SessionStore


class ConversationMemory:
    """Database-backed short-term memory with explicit turn and character budgets."""

    def __init__(self, store: SessionStore, max_turns: int = 6, max_chars: int = 12000):
        self.store = store
        self.max_turns = max(0, max_turns)
        self.max_chars = max(0, max_chars)

    def load(self, user_id: int, conversation_id: str) -> list[BaseMessage]:
        if self.max_turns == 0 or self.max_chars == 0:
            return []
        turns = self.store.recent_turns(user_id, conversation_id, self.max_turns)
        return self.build_messages(turns, self.max_chars)

    @staticmethod
    def build_messages(turns: list[dict[str, Any]], max_chars: int) -> list[BaseMessage]:
        if max_chars <= 0:
            return []

        selected: list[tuple[str, str]] = []
        used = 0
        for turn in reversed(turns):
            question = str(turn.get("question") or "").strip()
            answer = str(turn.get("answer") or "").strip()
            cost = len(question) + len(answer)

            if selected and used + cost > max_chars:
                break
            if cost > max_chars:
                remaining = max_chars
                question = question[:remaining]
                remaining -= len(question)
                answer = answer[:remaining]
                cost = len(question) + len(answer)

            selected.append((question, answer))
            used += cost
            if used >= max_chars:
                break

        messages: list[BaseMessage] = []
        for question, answer in reversed(selected):
            if question:
                messages.append(HumanMessage(content=question))
            if answer:
                messages.append(AIMessage(content=answer))
        return messages
