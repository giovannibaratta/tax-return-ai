from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from backend.db_manager import DatabaseManager, LocalDb
from backend.deliberation.models import EvidenceChunk
from backend.llm.reranker import RerankedResult
from src.ui.main_window import RegulationsTab
from src.ui.workers import SearchWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def db_instance(tmp_path):
    db_path = str(tmp_path / "test_search_db.db")
    vector_db_path = str(tmp_path / "test_vector_search_db.db")
    db = DatabaseManager(db_config=LocalDb(db_path=db_path, vector_db_path=vector_db_path))
    yield db
    db.close()


def test_search_worker_execution(db_instance: DatabaseManager, qapp):
    # Given: DB populated with a chunk and mock embedding runner
    db_instance.insert_chunk(
        document_name="tax_law.pdf",
        jurisdiction="italy",
        page_number=5,
        text_content="Exit tax on capital gains",
        chunk_index=0,
        embedding=[0.05] * 1024,
        document_sha="sha_tax_law",
        parent_chunk_id=1,
        parent_text_content="Full section on exit tax on capital gains for non-residents.",
    )

    mock_runner = MagicMock()
    mock_runner.embed.return_value = [0.05] * 1024

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, candidates, top_k=None, text_extractor=None: [
        RerankedResult(item=c, rerank_score=0.9) for c in candidates[:top_k]
    ]

    worker = SearchWorker(
        query="exit tax",
        db=db_instance,
        embedding_runner=mock_runner,
        reranker=mock_reranker,
    )

    results_received = []

    def on_results(res):
        results_received.extend(res)

    worker.results_found.connect(on_results)
    worker.error_occurred.connect(lambda e: print("ERROR:", e))

    # When: SearchWorker runs synchronously
    worker.run()

    # Then: Matching EvidenceChunk is emitted with parent context
    assert len(results_received) == 1
    chunk: EvidenceChunk = results_received[0]
    assert chunk.document_name == "tax_law.pdf"
    assert chunk.text_content == "Exit tax on capital gains"
    assert chunk.parent_text_content == "Full section on exit tax on capital gains for non-residents."


def test_regulations_tab_search_and_parent_chunk_ui(db_instance: DatabaseManager, qapp):
    # Given: RegulationsTab initialized with DB
    tab = RegulationsTab(db=db_instance)

    # Then: Search input, log, and parent chunk text area exist
    assert hasattr(tab, "_search_input")
    assert hasattr(tab, "_parent_chunk_text")
    assert tab._log.maximumHeight() == 130
    assert tab._parent_chunk_text.isReadOnly()
