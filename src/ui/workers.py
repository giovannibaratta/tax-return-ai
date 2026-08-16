"""Background worker threads for the Tax Return AI UI.

Each worker wraps a long-running backend operation and emits typed Qt signals
so the main thread can update the UI without blocking.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, cast

from pydantic_ai import Agent
from PySide6.QtCore import QObject, QThread, Signal

from backend.chat.models import ChatMessage
from backend.db_manager import DatabaseManager
from backend.ingestion.chunker import LateChunker
from backend.ingestion.ingest import ingest_document
from backend.llm.embedding_runner import BaseEmbeddingRunner
from backend.llm.reranker import BgeCrossEncoderReranker
from backend.utils.agents import SharedAgentDeps

if TYPE_CHECKING:
    from backend.chat.agent import ChatDeps
    from backend.chat.tax_filing_agent import TaxFilingDeps



class IngestionWorker(QThread):
    """Run document ingestion on a background thread.

    Emits:
        log_message: Informational or error line to display in the UI log.
        finished: Emitted when all files are processed. ``True`` if all
            succeeded; ``False`` if any file raised an exception.
    """

    log_message: Signal = Signal(str)
    ingest_finished: Signal = Signal(bool)

    def __init__(
        self,
        file_paths: list[str],
        jurisdiction: str,
        db: DatabaseManager,
        force: bool = False,
        force_ocr: bool = False,
        parent: QObject | None = None,
    ) -> None:
        """Initialise the worker.

        Args:
            file_paths: Absolute paths of PDF files to ingest.
            jurisdiction: Jurisdiction label ('italy' or 'ireland').
            db: Shared DatabaseManager instance (migrations already run).
            force: When True, re-ingest documents that are already present.
            force_ocr: When True, bypass local OCR cache and force raw parsing.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._file_paths = file_paths
        self._jurisdiction = jurisdiction
        self._db = db
        self._force = force
        self._force_ocr = force_ocr

    def run(self) -> None:
        """Execute ingestion and emit progress signals.

        Captures stdout-style prints by monkey-patching the built-in ``print``
        so that ingest_document output surfaces in the UI log widget.
        """
        import builtins

        original_print = builtins.print

        def _captured_print(*args: object, **kwargs: object) -> None:
            line = " ".join(str(a) for a in args)
            self.log_message.emit(line)

        builtins.print = _captured_print
        success = True
        try:
            chunker = LateChunker()
            for path in self._file_paths:
                self.log_message.emit(f"▶ Starting: {path}")
                ingest_document(
                    path,
                    self._db,
                    chunker,
                    self._jurisdiction,
                    force=self._force,
                    force_ocr=self._force_ocr,
                )
        except Exception:
            self.log_message.emit(traceback.format_exc())
            success = False
        finally:
            builtins.print = original_print
            self.ingest_finished.emit(success)


class SearchWorker(QThread):
    """Run semantic vector search on a background thread.

    Emits:
        results_found: Signal emitting list of EvidenceChunk search results.
        error_occurred: Signal emitting error string if search fails.
    """

    results_found: Signal = Signal(list)
    error_occurred: Signal = Signal(str)

    def __init__(
        self,
        query: str,
        db: DatabaseManager,
        embedding_runner: BaseEmbeddingRunner | None = None,
        jurisdiction: str | None = None,
        limit: int = 15,
        reranker: BgeCrossEncoderReranker | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialise the search worker.

        Args:
            query: The search query string.
            db: Shared DatabaseManager instance.
            embedding_runner: Optional pre-loaded BgeM3EmbeddingRunner instance.
            jurisdiction: Optional account country filter ('italy' or 'ireland').
            limit: Max results to return.
            reranker: Optional pre-loaded or mock BgeCrossEncoderReranker instance.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._query = query
        self._db = db
        self._embedding_runner = embedding_runner
        self._jurisdiction = jurisdiction
        self._limit = limit
        self._reranker = reranker

    def run(self) -> None:
        """Generate query embedding and perform two-stage RAG search with Cross-Encoder reranking."""
        try:
            from backend.deliberation.models import EvidenceChunk
            from backend.services.tax_services import query_tax_knowledge_action

            results = query_tax_knowledge_action(
                db=self._db,
                embedding_runner=self._embedding_runner,
                query_text=self._query,
                limit=self._limit,
                jurisdiction=self._jurisdiction,
                reranker=self._reranker,
            )
            self.results_found.emit(results)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class ChatWorker(QThread):
    """Run LLM chat agent turn asynchronously on a background thread.

    Emits:
        message_ready: Signal emitting (ChatMessage, list[ToolCallInfo], TokenUsageInfo) tuple.
        progress_updated: Signal emitting status string.
        approval_requested: Signal emitting (int, object) for request limit extension.
        error_occurred: Signal emitting error string if turn fails.
    """

    message_ready: Signal = Signal(object, list, object)
    progress_updated: Signal = Signal(str)
    approval_requested: Signal = Signal(int, object)
    error_occurred: Signal = Signal(str)

    def __init__(
        self,
        prompt: str,
        past_messages: list[ChatMessage],
        deps: SharedAgentDeps,
        agent: Agent[ChatDeps, str] | Agent[TaxFilingDeps, str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the chat worker.

        Args:
            prompt: User question/prompt string.
            past_messages: Historical ChatMessage instances in session.
            deps: SharedAgentDeps container (e.g. ChatDeps or TaxFilingDeps).
            agent: Optional pre-configured agent instance.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._prompt = prompt
        self._past_messages = past_messages
        self._deps = deps
        self._agent = agent

    def _request_limit_callback(self, current_limit: int) -> bool:
        """Thread-safe callback requesting user approval for extending request limit."""
        import threading

        event = threading.Event()
        result_container: list[bool] = [False]
        self.approval_requested.emit(current_limit, (event, result_container))
        event.wait()
        return result_container[0]

    def run(self) -> None:
        """Execute chat turn and emit results or error."""
        try:
            from backend.chat.agent import create_chat_agent, run_chat_turn_sync

            self._deps.on_progress = self.progress_updated.emit

            if self._agent is not None:
                assistant_msg, traces, usage_info = run_chat_turn_sync(
                    agent=cast(Agent[SharedAgentDeps, str], self._agent),
                    deps=self._deps,
                    prompt=self._prompt,
                    past_messages=self._past_messages,
                    request_limit_callback=self._request_limit_callback,
                )
            else:
                from backend.chat.agent import ChatDeps

                default_agent = create_chat_agent()
                chat_deps = (
                    self._deps
                    if isinstance(self._deps, ChatDeps)
                    else ChatDeps(
                        db=self._deps.db,
                        embedding_runner=self._deps.embedding_runner,
                        on_progress=self._deps.on_progress,
                    )
                )
                assistant_msg, traces, usage_info = run_chat_turn_sync(
                    agent=default_agent,
                    deps=chat_deps,
                    prompt=self._prompt,
                    past_messages=self._past_messages,
                    request_limit_callback=self._request_limit_callback,
                )
            self.message_ready.emit(assistant_msg, traces, usage_info)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())
