from collections.abc import Generator
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import FinancialRecord, StagedFinancialRecord
from tests.utils import insert_financial_record


@pytest.fixture
def db() -> Generator[DatabaseManager, None, None]:
    """Fixture providing an in-memory database manager instance."""
    manager = DatabaseManager(MemoryDb())
    yield manager
    manager.close()


def test_staged_record_validation() -> None:
    """Test required field checks on StagedFinancialRecord."""
    # Given: An incomplete staged record missing action and quantity
    incomplete_staged = StagedFinancialRecord(
        provider="revolut",
        account_country="ireland",
        event_timestamp=datetime(2025, 5, 10, 12, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        unit_price=Decimal("150.00"),
        total_amount=Decimal("150.00"),
        tax_year=2025,
    )

    # When: Checking approvable state and missing fields
    missing_fields = incomplete_staged.get_missing_fields()
    is_app = incomplete_staged.is_approvable()

    # Then: Record is not approvable and missing fields list action and quantity
    assert is_app is False
    assert "action" in missing_fields
    assert "quantity" in missing_fields


def test_approve_staged_record_workflow(db: DatabaseManager) -> None:
    """Test approving a valid staged record into financial_records."""
    # Given: A valid staged transaction inserted into the database
    staged = StagedFinancialRecord(
        provider="degiro",
        source_file_name="statement.pdf",
        source_file_sha="abc123sha",
        event_timestamp=datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="NVDA",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("120.00"),
        currency="EUR",
        total_amount=Decimal("1200.00"),
        tax_year=2025,
        account_country="ireland",
        verification_status="pending_approval",
    )
    inserted_staged = db.insert_staged_record(staged)
    assert inserted_staged.id is not None

    # When: Approving the staged record
    success, msg, financial_rec, dups = db.approve_staged_record(inserted_staged.id)

    # Then: Record is approved, moved to financial_records, and staged record status is updated
    assert success is True
    assert msg == "approved"
    assert financial_rec is not None
    assert financial_rec.symbol == "NVDA"
    assert financial_rec.quantity == Decimal("10.0")

    staged_reloaded = db.get_staged_record_by_id(inserted_staged.id)
    assert staged_reloaded is not None
    assert staged_reloaded.verification_status == "approved"
    assert staged_reloaded.approved_financial_record_id == financial_rec.id

    ledger_records = db.get_financial_records()
    assert len(ledger_records) == 1
    assert ledger_records[0].id == financial_rec.id


def test_duplicate_heuristic_warning(db: DatabaseManager) -> None:
    """Test duplicate detection heuristic when approving duplicate transaction."""
    # Given: An existing approved record in financial_records
    approved_existing = FinancialRecord(
        provider="degiro",
        source_file_sha="abc123sha",
        event_timestamp=datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="NVDA",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("120.00"),
        currency="EUR",
        total_amount=Decimal("1200.00"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(db, approved_existing)

    # And: A new staged record matching the same symbol, quantity, action, and calendar day
    staged_duplicate = StagedFinancialRecord(
        provider="degiro",
        source_file_sha="abc123sha2",
        event_timestamp=datetime(2025, 6, 15, 16, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="NVDA",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("120.00"),
        currency="EUR",
        total_amount=Decimal("1200.00"),
        tax_year=2025,
        account_country="ireland",
        verification_status="pending_approval",
    )
    staged_inserted = db.insert_staged_record(staged_duplicate)
    assert staged_inserted.id is not None

    # When: Attempting normal approval without force_duplicate
    success, msg, created_rec, dups = db.approve_staged_record(staged_inserted.id, force_duplicate=False)

    # Then: Approval is blocked with potential_duplicate warning
    assert success is False
    assert msg == "potential_duplicate"
    assert len(dups) == 1
    assert dups[0].id == approved_existing.id

    # When: Forcing approval after user confirmation
    forced_success, forced_msg, forced_rec, _ = db.approve_staged_record(staged_inserted.id, force_duplicate=True)

    # Then: Approval succeeds and new financial record is created
    assert forced_success is True
    assert forced_rec is not None
    ledger_records = db.get_financial_records()
    assert len(ledger_records) == 2


def test_openfigi_detected_field_and_approval(db: DatabaseManager) -> None:
    """Test storing openfigi_detected field in staged record and approving into ledger."""
    # Given: A staged record with openfigi_detected populated
    staged = StagedFinancialRecord(
        provider="directa",
        source_file_name="directa.pdf",
        source_file_sha="abc123sha",
        event_timestamp=datetime(2025, 7, 4, 10, 30, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="ENI",
        isin="IE00BFWXDV39",
        openfigi_detected="ENI",
        action="buy",
        quantity=Decimal("100.0"),
        unit_price=Decimal("14.50"),
        currency="EUR",
        total_amount=Decimal("1455.00"),
        tax_year=2025,
        account_country="italy",
        verification_status="pending_approval",
    )
    inserted = db.insert_staged_record(staged)
    assert inserted.id is not None
    assert inserted.openfigi_detected == "ENI"

    # When: Approving the record into financial_records
    success, msg, created_rec, _ = db.approve_staged_record(inserted.id)

    # Then: FinancialRecord in ledger preserves openfigi_detected
    assert success is True
    assert created_rec is not None
    assert created_rec.openfigi_detected == "ENI"
    assert created_rec.symbol == "ENI"
