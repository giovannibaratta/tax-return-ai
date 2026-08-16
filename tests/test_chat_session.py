"""Tests for SessionStore and ChatSession local JSON storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

from backend.chat.models import (
    AssistantChatMessage,
    UserChatMessage,
)
from backend.chat.session_store import SessionStore
from backend.utils.agents import (
    DocumentChunkResource,
    FinancialRecordResource,
    ToolCallInfo,
)


def test_session_store_crud() -> None:
    # Given: a temporary directory for session storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SessionStore(sessions_dir=tmp_dir)

        # When: a new session is created with auto_save=True
        session = store.create_session(title="Tax Inquiry 2026", auto_save=True)
        session_id = session.id

        # Then: the session file should exist on disk
        assert (Path(tmp_dir) / f"{session_id}.json").exists()
        assert len(store.list_sessions()) == 1

        # When: messages and tool calls are added to session
        msg_user = UserChatMessage(content="What is the exit tax rate in Ireland?")
        tool_call = ToolCallInfo(
            tool_name="query_tax_knowledge",
            args={"query_text": "exit tax Ireland"},
            result_summary="Found 1 chunk",
            resources=[
                DocumentChunkResource(
                    document_name="tax_manual.pdf",
                    jurisdiction="ireland",
                    chunk_id=42,
                    snippet="41% exit tax rate...",
                )
            ],
        )
        msg_assistant = AssistantChatMessage(
            content="The exit tax rate in Ireland is 41%.",
            tool_calls=[tool_call],
        )

        session.messages.extend([msg_user, msg_assistant])
        store.save_session(session)

        # Then: reloading session should preserve full structure and tool calls
        loaded = store.load_session(session_id)
        assert loaded is not None
        assert loaded.title == "Tax Inquiry 2026"
        assert len(loaded.messages) == 2
        assert isinstance(loaded.messages[1], AssistantChatMessage)
        chunk_res = loaded.messages[1].tool_calls[0].resources[0]
        assert isinstance(chunk_res, DocumentChunkResource)
        assert chunk_res.chunk_id == 42

        # When: session is deleted
        deleted = store.delete_session(session_id)

        # Then: deletion should return True and store should be empty
        assert deleted is True
        assert len(store.list_sessions()) == 0
        assert store.load_session(session_id) is None


def test_session_store_financial_record_resource() -> None:
    # Given: Session with Assistant message containing both DocumentChunkResource and FinancialRecordResource
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SessionStore(sessions_dir=tmp_dir)
        session = store.create_session(title="Financial Audit")

        chunk_res = DocumentChunkResource(
            document_name="manual.pdf",
            jurisdiction="italy",
            chunk_id=10,
        )
        fin_res = FinancialRecordResource(
            record_id=31,
        )
        tool_call = ToolCallInfo(
            tool_name="filter_financial_records",
            args={"logic": "AND"},
            result_summary="Found 1 record",
            resources=[chunk_res, fin_res],
        )
        msg_assistant = AssistantChatMessage(
            content="Audit completed.",
            tool_calls=[tool_call],
        )
        session.messages.append(msg_assistant)
        store.save_session(session)

        # When: Reloading session from disk
        loaded = store.load_session(session.id)

        # Then: Loaded session preserves distinct resource types
        assert loaded is not None
        loaded_msg = loaded.messages[0]
        assert isinstance(loaded_msg, AssistantChatMessage)
        res_list = loaded_msg.tool_calls[0].resources
        assert len(res_list) == 2
        assert isinstance(res_list[0], DocumentChunkResource)
        assert isinstance(res_list[1], FinancialRecordResource)
        assert res_list[1].record_id == 31


def test_session_store_in_memory_until_saved() -> None:
    # Given: A session store with auto_save=False
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SessionStore(sessions_dir=tmp_dir)

        # When: Creating session without auto_save
        session = store.create_session(title="Pending Chat", auto_save=False)

        # Then: File is not created on disk
        assert not (Path(tmp_dir) / f"{session.id}.json").exists()
        assert len(store.list_sessions()) == 0

        # When: Explicitly saving
        store.save_session(session)

        # Then: File is created on disk
        assert (Path(tmp_dir) / f"{session.id}.json").exists()
        assert len(store.list_sessions()) == 1


def test_tax_filing_session_store() -> None:
    # Given: A specialized session store for IrishTaxFilingSession
    from src.jurisdiction.ireland.tax_form_models import IrishForm11State, IrishTaxFilingSession

    with tempfile.TemporaryDirectory() as tmp_dir:
        filing_store = SessionStore(sessions_dir=tmp_dir, session_cls=IrishTaxFilingSession)

        # When: Saving an Irish tax filing session with form state
        session = filing_store.create_session(
            title="Return 2025",
            form_state=IrishForm11State(tax_year=2025),
            auto_save=False,
        )
        filing_store.save_session(session)

        # Then: Loaded session preserves typed form_state
        loaded = filing_store.load_session(session.id)
        assert loaded is not None
        assert isinstance(loaded, IrishTaxFilingSession)
        assert loaded.form_state is not None
        assert loaded.form_state.tax_year == 2025
        assert loaded.form_state.form_type == "form11"
        
