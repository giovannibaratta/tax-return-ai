"""Tests for VoterDiffMergeDialog and Candidate Record Retention."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication
from sqlmodel import SQLModel

from backend.consensus_models import ConsensusLog, TransactionExtractionItem
from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import FinancialRecord
from backend.domain_models import AssetType, TransactionAction
from src.ui.voter_diff_dialog import VoterDiffMergeDialog
from tests.utils import insert_financial_record


@pytest.fixture(scope="session")
def qapp():
    """Share a single QApplication instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_db():
    db = DatabaseManager(MemoryDb())
    SQLModel.metadata.create_all(db.engine)
    yield db
    db.close()


def test_voter_diff_dialog_populates_and_approves(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: A financial record with consensus log containing candidate voter extractions
    v1_item = TransactionExtractionItem(
        event_date=datetime(2025, 1, 1, 10, 0),
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        action=TransactionAction.BUY,
        quantity=Decimal("10.0"),
        unit_price=Decimal("150.0"),
        total_amount=Decimal("1500.0"),
        currency="USD",
        fx_rate=Decimal("1.0"),
        fees=Decimal("0.0"),
    )
    v2_item = TransactionExtractionItem(
        event_date=datetime(2025, 1, 1, 10, 0),
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        action=TransactionAction.BUY,
        quantity=Decimal("12.0"),  # Mismatch
        unit_price=Decimal("150.0"),
        total_amount=Decimal("1800.0"),
        currency="USD",
        fx_rate=Decimal("1.0"),
        fees=Decimal("0.0"),
    )
    c_log = ConsensusLog(
        version="1.0",
        error="Field value mismatch",
        raw_voter_1_records=[v1_item],
        raw_voter_2_records=[v2_item],
        raw_voter_3_records=[v1_item],
    )

    rec = FinancialRecord(
        provider="interactive_brokers",
        source_file_name="ibkr_test.pdf",
        event_timestamp=datetime(2025, 1, 1, 10, 0),
        asset_type="stock",
        symbol="AAPL",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("150.0"),
        total_amount=Decimal("1500.0"),
        currency="USD",
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1500.0"),
        tax_year=2025,
        account_country="ireland",
        verification_status="escalated_to_user",
        consensus_log=c_log.model_dump_json(),
        fees=Decimal("0.0"),
    )
    inserted = insert_financial_record(test_db, rec)

    # When: Opening VoterDiffMergeDialog and approving merged record
    dialog = VoterDiffMergeDialog(inserted, db=test_db)
    assert dialog.table.rowCount() == 12

    # Click Apply V2 for quantity row
    dialog._apply_voter_all(2)
    with (
        patch("src.ui.voter_diff_dialog.QMessageBox.information"),
        patch("src.ui.voter_diff_dialog.QMessageBox.critical") as mock_crit,
    ):
        dialog._approve_merged_record()
        if mock_crit.called:
            raise RuntimeError(f"QMessageBox.critical called: {mock_crit.call_args}")

    # Then: Financial record in DB updated with V2 quantity and status approved
    assert inserted.id is not None
    updated = test_db.get_financial_record_by_id(inserted.id)
    assert updated is not None
    assert updated.verification_status == "approved"
    assert updated.quantity == Decimal("12.0")
