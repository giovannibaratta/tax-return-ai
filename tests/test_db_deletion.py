from datetime import datetime
from pathlib import Path

import pytest

from backend.db_manager import DatabaseManager, LocalDb
from backend.db_models import FinancialRecord
from backend.domain_models import AssetType, ConfidenceLevel, SourceType, TransactionAction
from tests.utils import insert_financial_record


@pytest.fixture
def db_instance(tmp_path: Path):
    db_path = str(tmp_path / "test_db.db")
    db = DatabaseManager(db_config=LocalDb(db_path=db_path, vector_db_path=str(Path(db_path).parent / "vector.db")))
    yield db
    db.close()


def test_delete_all_chunks(db_instance: DatabaseManager):
    # Given: DB populated with chunks, vectors, and a financial record
    db_instance.insert_chunk(
        document_name="test_doc.pdf",
        jurisdiction="italy",
        page_number=1,
        text_content="Sample text content for chunk 0",
        chunk_index=0,
        embedding=[0.1] * 1024,
        document_sha="abc123sha",
        source_type=SourceType.REGULATION,
        confidence_level=ConfidenceLevel.HIGH,
    )
    db_instance.insert_chunk(
        document_name="test_doc.pdf",
        jurisdiction="italy",
        page_number=2,
        text_content="Sample text content for chunk 1",
        chunk_index=1,
        embedding=[0.2] * 1024,
        document_sha="abc123sha",
        source_type=SourceType.REGULATION,
        confidence_level=ConfidenceLevel.HIGH,
    )
    financial_rec = insert_financial_record(db_instance,
        FinancialRecord(
            account_country="italy",
            tax_year=2025,
            provider="degiro",
            asset_type=AssetType.STOCK,
            action=TransactionAction.BUY,
            event_timestamp=datetime.now(),
            source_file_name="report.pdf",
            source_file_sha="sha_financial",
        )
    )

    assert len(db_instance.get_ingested_documents()) == 1

    # When: delete_all_chunks is invoked
    deleted_count = db_instance.delete_all_chunks()

    # Then: Chunks and vectors are cleared, but financial records remain
    assert deleted_count == 2
    assert len(db_instance.get_ingested_documents()) == 0
    assert financial_rec.id is not None
    assert db_instance.get_financial_record_by_id(financial_rec.id) is not None
