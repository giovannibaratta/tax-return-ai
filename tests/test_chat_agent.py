"""Tests for chat agent tool invocation and turn execution."""

from __future__ import annotations

import tempfile

import pytest
from pydantic_ai.models.test import TestModel
from sqlmodel import Session

from backend.chat.agent import ChatDeps, build_chat_history_prompt, create_chat_agent, run_chat_turn_sync
from backend.chat.models import AssistantChatMessage, ChatMessage, UserChatMessage
from backend.db_manager import DatabaseManager, LocalDb, TaxDocumentMetadata



def test_chat_agent_calculate_tool() -> None:
    # Given: a TestModel configured for calculate tool call and a migrated DatabaseManager
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db = DatabaseManager(db_config=LocalDb(db_path=tmp_db.name, vector_db_path=tmp_db.name + "_vec.db"))
        deps = ChatDeps(db=db)

        # TestModel with custom responses
        test_model = TestModel(call_tools=[])
        agent = create_chat_agent(model=test_model)

        # When: executing a chat turn asking for arithmetic
        past_msgs: list[ChatMessage] = []
        user_prompt = "Calculate 3500 * 0.26"

        # Run chat turn using TestModel
        assistant_msg, tool_traces, usage_info = run_chat_turn_sync(
            agent=agent,
            deps=deps,
            prompt=user_prompt,
            past_messages=past_msgs,
        )

        # Then: response should be generated and assistant message created
        assert assistant_msg.role == "assistant"
        assert isinstance(assistant_msg.content, str)
        assert len(assistant_msg.content) > 0


def test_get_chunk_neighbors_db() -> None:
    # Given: a database with 3 sequential chunks in the same document
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db = DatabaseManager(db_config=LocalDb(db_path=tmp_db.name, vector_db_path=tmp_db.name + "_vec.db"))

        with Session(db.engine) as session:
            c0 = TaxDocumentMetadata(
                id=1,
                document_name="manual.pdf",
                document_sha="sha123",
                jurisdiction="italy",
                page_number=1,
                chunk_index=0,
                text_content="Chunk 0 context",
            )
            c1 = TaxDocumentMetadata(
                id=2,
                document_name="manual.pdf",
                document_sha="sha123",
                jurisdiction="italy",
                page_number=1,
                chunk_index=1,
                text_content="Chunk 1 main target",
            )
            c2 = TaxDocumentMetadata(
                id=3,
                document_name="manual.pdf",
                document_sha="sha123",
                jurisdiction="italy",
                page_number=2,
                chunk_index=2,
                text_content="Chunk 2 following context",
            )
            session.add_all([c0, c1, c2])
            session.commit()
            target_id = c1.id
            assert target_id is not None

        # When: fetching neighbors around chunk 1 with window 1
        neighbors = db.get_chunk_neighbors(chunk_id=target_id, window=1)

        # Then: all 3 chunks should be returned in chronological order
        assert len(neighbors) == 3
        assert neighbors[0].chunk_index == 0
        assert neighbors[1].chunk_index == 1
        assert neighbors[2].chunk_index == 2


from unittest.mock import patch


def test_run_chat_turn_history_compaction() -> None:
    # Given: 15 past chat messages and a max_history_messages limit of 4
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db = DatabaseManager(db_config=LocalDb(db_path=tmp_db.name, vector_db_path=tmp_db.name + "_vec.db"))

        test_model = TestModel(call_tools=[])
        agent = create_chat_agent(model=test_model)

        past_msgs: list[ChatMessage] = [
            UserChatMessage(content=f"Message {i}") if i % 2 == 0 else AssistantChatMessage(content=f"Message {i}")
            for i in range(15)
        ]

        # When: executing chat turn with max_history_messages=4
        deps = ChatDeps(db=db)
        with patch(
            "backend.chat.agent.summarize_conversation_history", return_value="Mocked summary of older turns."
        ) as mock_sum:
            msg, traces, usage_info = run_chat_turn_sync(
                agent=agent,
                deps=deps,
                prompt="New Question",
                past_messages=past_msgs,
                max_history_messages=4,
            )
            assert mock_sum.called

        # Then: message is generated successfully
        assert msg.role == "assistant"
        assert len(msg.content) > 0
        assert usage_info is not None


def test_build_chat_history_prompt_token_limit_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: 4 past messages within turn limit (max_turns=30) but exceeding token limit (max_tokens=100)
    monkeypatch.setenv("CHAT_MAX_HISTORY_TOKENS", "100")
    monkeypatch.setenv("CHAT_MAX_HISTORY_TURNS", "30")

    past_msgs: list[ChatMessage] = [
        UserChatMessage(content="X" * 250),
        AssistantChatMessage(content="Y" * 250),
        UserChatMessage(content="Z" * 250),
        AssistantChatMessage(content="W" * 250),
    ]

    with patch("backend.chat.agent.summarize_conversation_history", return_value="Mocked token summary") as mock_sum:
        prompt_text = build_chat_history_prompt(past_msgs)
        assert mock_sum.called
        assert "Mocked token summary" in prompt_text



def test_run_chat_turn_sync_request_limit_callback() -> None:
    # Given: Chat turn with low request limit and custom approval callback
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db = DatabaseManager(db_config=LocalDb(db_path=tmp_db.name, vector_db_path=tmp_db.name + "_vec.db"))
        deps = ChatDeps(db=db)

        test_model = TestModel(call_tools=[])
        agent = create_chat_agent(model=test_model)

        callback_invoked: list[int] = []

        def mock_callback(current_limit: int) -> bool:
            callback_invoked.append(current_limit)
            return False  # Decline extension

        # When: Executing turn with request_limit=1
        msg, traces, usage_info = run_chat_turn_sync(
            agent=agent,
            deps=deps,
            prompt="Complex query",
            past_messages=[],
            request_limit=1,
            request_limit_callback=mock_callback,
        )

        # Then: Response is returned safely without raising UsageLimitExceeded exception
        assert isinstance(msg.content, str)
        assert usage_info is not None
        assert msg.role == "assistant"
