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


def test_generic_voter_diff_dialog_pure_selection(qapp: QApplication) -> None:
    # Given: Field specs for generic dialog
    from src.ui.voter_diff_dialog import GenericVoterDiffDialog, VoterDiffField

    fields = [
        VoterDiffField(
            key="employer", label="Employer", val_v1="Acme", val_v2="Acme Inc", val_v3="Acme", initial_merged_val=""
        ),
        VoterDiffField(
            key="gross_pay",
            label="Gross Pay",
            val_v1="50000.00",
            val_v2="50000.00",
            val_v3="50000.00",
            initial_merged_val="50000.00",
            is_numeric=True,
        ),
    ]

    dialog = GenericVoterDiffDialog(
        title="Test Diff",
        header_html="<b>Test</b>",
        fields=fields,
    )
    assert dialog.table.rowCount() == 2

    # When: Applying V2 to all fields and clicking accept
    dialog._apply_voter_all(2)
    dialog._on_accept_clicked()

    # Then: Merged values dictionary returns V2 values
    merged = dialog.get_merged_values()
    assert merged["employer"] == "Acme Inc"
    assert merged["gross_pay"] == "50000.00"


def test_income_voter_diff_dialog_extract_and_approve(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: A staged tax income record with voter outputs
    from backend.domain_models import IrishEmploymentDetailSummaryPayload, StrictStagedTaxIncomeRecord
    from src.ui.income_voter_diff_dialog import IncomeVoterDiffDialog

    p1 = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Google Ireland",
        gross_pay_eur=Decimal("80000.00"),
        income_tax_paid_eur=Decimal("20000.00"),
        usc_paid_eur=Decimal("3500.00"),
        prsi_paid_eur=Decimal("3200.00"),
    )
    p2 = p1.model_copy(update={"gross_pay_eur": Decimal("82000.00")})
    p3 = p1.model_copy(update={"gross_pay_eur": Decimal("82000.00")})

    staged = StrictStagedTaxIncomeRecord(
        tax_year=2025,
        jurisdiction="ireland",
        income_type="irish_employment_detail_summary",
        source_file_name="google_eds.pdf",
        payload=None,
        voter_outputs=[p1, p2, p3],
        verification_status="escalated_to_user",
        created_at=datetime.now(),
    )
    staged_id = test_db.insert_staged_tax_income_record(staged)
    staged.id = staged_id

    # When: Opening IncomeVoterDiffDialog and applying V2
    dialog = IncomeVoterDiffDialog(staged, db=test_db)
    assert dialog.table.rowCount() == 18

    dialog._apply_voter_all(2)

    with (
        patch("src.ui.income_voter_diff_dialog.QMessageBox.information"),
        patch("src.ui.income_voter_diff_dialog.QMessageBox.critical") as mock_crit,
    ):
        dialog._approve_merged_record()
        if mock_crit.called:
            raise RuntimeError(f"QMessageBox.critical called: {mock_crit.call_args}")

    # Then: Staged record promoted to ledger with V2 gross pay
    approved_list = test_db.get_tax_income_records(tax_year=2025)
    assert len(approved_list) == 1
    assert approved_list[0].payload.gross_pay_eur == Decimal("82000.00")


def test_income_record_details_dialog(qapp: QApplication) -> None:
    # Given: An approved Irish EDS tax income record
    from backend.domain_models import (
        IrishEmploymentDetailSummaryPayload,
        IrishPRSIClassEntry,
        StrictTaxIncomeRecord,
    )
    from src.ui.income_details_dialog import IncomeRecordDetailsDialog

    payload = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Meta Ireland",
        employer_registration_number="1234567T",
        employment_id="EMP-01",
        gross_pay_eur=Decimal("95000.00"),
        pay_for_income_tax_eur=Decimal("95000.00"),
        income_tax_paid_eur=Decimal("25000.00"),
        taxable_benefits_eur=Decimal("1200.00"),
        pay_for_usc_eur=Decimal("96200.00"),
        usc_paid_eur=Decimal("4200.00"),
        prsi_paid_eur=Decimal("3800.00"),
        employer_prsi_paid_eur=Decimal("10500.00"),
        prsi_classes=[
            IrishPRSIClassEntry(prsi_class="A1", insurable_weeks=40),
            IrishPRSIClassEntry(prsi_class="M", insurable_weeks=0),
        ],
    )
    rec = StrictTaxIncomeRecord(
        id=99,
        tax_year=2025,
        jurisdiction="ireland",
        income_type="irish_employment_detail_summary",
        payload=payload,
        created_at=datetime.now(),
    )

    # When: Opening IncomeRecordDetailsDialog
    dialog = IncomeRecordDetailsDialog(rec)

    # Then: Dialog initializes properly with all cards
    assert dialog.windowTitle() == "📑 Approved Tax Income Record #99 (2025 - Ireland)"
