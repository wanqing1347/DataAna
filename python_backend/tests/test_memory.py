from langchain_core.messages import AIMessage, HumanMessage

from app.memory import ConversationMemory


def test_memory_keeps_recent_turns_in_chronological_order():
    turns = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"},
    ]
    messages = ConversationMemory.build_messages(turns, max_chars=8)

    assert [type(x) for x in messages] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
        AIMessage,
    ]
    assert [x.content for x in messages] == ["q2", "a2", "q3", "a3"]


def test_memory_clips_single_oversized_latest_turn():
    turns = [{"question": "123456", "answer": "abcdef"}]
    messages = ConversationMemory.build_messages(turns, max_chars=8)

    assert [x.content for x in messages] == ["123456", "ab"]
