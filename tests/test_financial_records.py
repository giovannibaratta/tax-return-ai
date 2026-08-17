"""Unit tests for FinancialRecord models, StrictFinancialRecord validation, and DatabaseManager queries."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import FinancialRecord
from backend.domain_models import (
    AssetType,
    BaseStrictRecord,
    DividendRecord,
    IrishEmploymentDetailSummaryPayload,
    StrictTaxIncomeRecord,
    TradeRecord,
    TransactionAction,
    VerificationStatus,
)
from tests.utils import insert_financial_record


@pytest.fixture
def db_manager():
    mgr = DatabaseManager(MemoryDb())
    yield mgr
    mgr.close()


def test_strict_financial_record_validation():
    # Given: An incomplete raw record missing required fields
    incomplete_rec = FinancialRecord(
        id=1,
        provider="directa",
        event_timestamp=None,  # missing timestamp
        asset_type="stock",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        total_amount=Decimal("1000"),
        tax_year=2024,
        account_country="italy",
    )

    # When/Then: Attempting to convert incomplete raw record raises ValueError
    with pytest.raises(ValueError):
        BaseStrictRecord.from_raw(incomplete_rec)

    # Given: A complete raw record with all required financial fields
    complete_rec = FinancialRecord(
        id=2,
        provider="directa",
        source_file_sha="sha_complete",
        event_timestamp=datetime(2024, 5, 10, 14, 30, tzinfo=timezone.utc),
        ingestion_timestamp=datetime(2024, 5, 10, 15, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        action="buy",
        quantity=Decimal("5.0"),
        unit_price=Decimal("180.0"),
        currency="USD",
        fees=Decimal("2.0"),
        total_amount=Decimal("902.0"),
        fx_rate=Decimal("0.92"),
        local_total_amount=Decimal("829.84"),
        tax_year=2024,
        account_country="italy",
        verification_status="pending_verification",
    )

    # When: Converting complete raw record to StrictFinancialRecord
    strict_complete = BaseStrictRecord.from_raw(complete_rec)

    # Then: Strict object is parsed with correct typed enums and local total EUR calculation
    assert isinstance(strict_complete, TradeRecord)
    assert strict_complete is not None
    assert strict_complete.id == 2
    assert strict_complete.asset_type == AssetType.STOCK
    assert strict_complete.action == TransactionAction.BUY
    assert strict_complete.total_amount == Decimal("902.0")
    assert strict_complete.local_total_amount == Decimal("829.84")


def test_update_financial_record_in_db(db_manager: DatabaseManager):
    # Given: An initial raw record in DB
    record = FinancialRecord(
        provider="manual",
        source_file_sha="sha_manual",
        event_timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        action="buy",
        total_amount=Decimal("500.00"),
        tax_year=2024,
        account_country="ireland",
        verification_status="escalated_to_user",
    )
    inserted = insert_financial_record(db_manager, record)
    assert inserted.id is not None

    # When: Updating fields (adding quantity, unit price, and setting status to approved)
    inserted.quantity = Decimal("10")
    inserted.unit_price = Decimal("50.00")
    inserted.currency = "EUR"
    inserted.fx_rate = Decimal("1.0")
    inserted.local_total_amount = Decimal("500.00")
    inserted.fees = Decimal("0.0")
    inserted.verification_status = "approved"

    strict_record = BaseStrictRecord.from_raw(inserted)
    updated = db_manager.update_strict_financial_record(strict_record)

    # Then: Database contains updated record values
    assert updated.id is not None
    fetched = db_manager.get_strict_financial_record(updated.id)
    assert fetched is not None
    assert isinstance(fetched, TradeRecord)
    assert fetched.quantity == Decimal("10")
    assert fetched.unit_price == Decimal("50.00")
    assert fetched.verification_status == VerificationStatus.APPROVED
    assert fetched.id == updated.id


def test_discriminated_strict_record_from_raw():
    # Given: A trade record with buy action
    raw_trade = FinancialRecord(
        id=10,
        provider="directa",
        event_timestamp=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("150.00"),
        currency="USD",
        fees=Decimal("1.50"),
        total_amount=Decimal("1501.50"),
        fx_rate=Decimal("0.92"),
        local_total_amount=Decimal("1381.38"),
        tax_year=2025,
        account_country="italy",
        verification_status="approved",
    )

    # Given: A dividend record without quantity or unit_price
    raw_dividend = FinancialRecord(
        id=11,
        provider="directa",
        event_timestamp=datetime(2025, 5, 16, 0, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="REY",
        isin="IT0005282865",
        action="dividend",
        quantity=None,
        unit_price=None,
        currency="EUR",
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("5.75"),
        fees=Decimal("0.0"),
        total_amount=Decimal("5.75"),
        tax_year=2025,
        account_country="italy",
        verification_status="approved",
    )

    # When: Converting raw records to strict models
    strict_trade = BaseStrictRecord.from_raw(raw_trade)
    strict_dividend = BaseStrictRecord.from_raw(raw_dividend)

    # Then: Strict trade is a TradeRecord with quantity and unit_price
    assert isinstance(strict_trade, TradeRecord)
    assert strict_trade.action == TransactionAction.BUY
    assert strict_trade.quantity == Decimal("10")
    assert strict_trade.unit_price == Decimal("150.00")

    # Then: Strict dividend is a DividendRecord without quantity or unit_price attributes
    assert isinstance(strict_dividend, DividendRecord)
    assert strict_dividend.action == TransactionAction.DIVIDEND
    assert strict_dividend.total_amount == Decimal("5.75")
    assert not hasattr(strict_dividend, "quantity") or getattr(strict_dividend, "quantity", None) is None
    assert not hasattr(strict_dividend, "unit_price") or getattr(strict_dividend, "unit_price", None) is None


def test_tax_income_records_db_persistence(db_manager: DatabaseManager) -> None:
    # Given: A validated Irish EDS payload and domain model
    payload = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Google Ireland Ltd",
        employer_registration_number="1234567T",
        employment_id="1",
        gross_pay_eur=Decimal("95000.00"),
        income_tax_paid_eur=Decimal("28000.00"),
        usc_paid_eur=Decimal("4200.00"),
        prsi_paid_eur=Decimal("3800.00"),
        prsi_class="A1",
        prsi_weeks=52,
    )
    strict_record = StrictTaxIncomeRecord(
        tax_year=2025,
        jurisdiction="ireland",
        income_type=payload.income_type,
        source_document_sha="abcd1234efgh5678",
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )

    # When: Inserted and retrieved from DB
    rec_id = db_manager.insert_tax_income_record(strict_record)
    retrieved = db_manager.get_tax_income_records(tax_year=2025, jurisdiction="ireland")

    # Then: Record is preserved with strongly-typed payload
    assert rec_id > 0
    assert len(retrieved) == 1
    assert retrieved[0].tax_year == 2025
    assert retrieved[0].jurisdiction == "ireland"
    assert isinstance(retrieved[0].payload, IrishEmploymentDetailSummaryPayload)
    assert retrieved[0].payload.employer_name == "Google Ireland Ltd"
    assert retrieved[0].payload.gross_pay_eur == Decimal("95000.00")
    assert retrieved[0].payload.prsi_weeks == 52
