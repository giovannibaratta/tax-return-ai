"""Unit tests for financial record ID lookup and filtering tools in DatabaseManager and tax_tools."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import FinancialRecord
from backend.domain_models import BaseStrictRecord, TradeRecord
from backend.llm.embedding_runner import BaseEmbeddingRunner
from tests.utils import insert_financial_record


class DummyEmbeddingRunner(BaseEmbeddingRunner):
    def embed(self, text: str) -> list[float]:
        return [0.0] * 1024


@pytest.fixture
def test_db():
    """Fixture providing a DatabaseManager populated with sample financial records."""
    mgr = DatabaseManager(MemoryDb())

    r1 = FinancialRecord(
        provider="directa",
        source_file_name="statement1.pdf",
        source_file_sha="sha_statement1",
        event_timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        ingestion_timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("50"),
        unit_price=Decimal("150.00"),
        fees=Decimal("0.00"),
        total_amount=Decimal("7500.00"),
        currency="EUR",
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("7500.00"),
        tax_year=2024,
        account_country="italy",
        verification_status="approved",
    )
    r2 = FinancialRecord(
        provider="interactive_brokers",
        source_file_name="statement2.pdf",
        source_file_sha="sha_statement2",
        event_timestamp=datetime(2024, 3, 20, 14, 30, tzinfo=timezone.utc),
        ingestion_timestamp=datetime(2024, 3, 20, 14, 30, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFMXXD85",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("80.00"),
        fees=Decimal("0.00"),
        total_amount=Decimal("800.00"),
        currency="EUR",
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("800.00"),
        tax_year=2024,
        account_country="italy",
        verification_status="approved",
    )
    r3 = FinancialRecord(
        provider="directa",
        source_file_name="statement3.pdf",
        source_file_sha="sha_statement3",
        event_timestamp=datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc),
        ingestion_timestamp=datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="MSFT",
        isin="US5949181045",
        action="buy",
        quantity=Decimal("5"),
        unit_price=Decimal("400.00"),
        fees=Decimal("0.00"),
        total_amount=Decimal("2000.00"),
        currency="EUR",
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("2000.00"),
        tax_year=2024,
        account_country="italy",
        verification_status="approved",
    )

    insert_financial_record(mgr, r1)
    insert_financial_record(mgr, r2)
    insert_financial_record(mgr, r3)

    return mgr


def test_get_financial_record_by_id(test_db: DatabaseManager):
    # Given: DB populated with 3 records

    # When: Fetching record ID 1
    rec = test_db.get_financial_record_by_id(1)

    # Then: Apple stock record is returned
    assert rec is not None
    assert rec.symbol == "AAPL"
    assert rec.quantity == Decimal("50")

    # When: Fetching non-existent record ID
    missing = test_db.get_financial_record_by_id(999)

    # Then: None is returned
    assert missing is None


def test_filter_by_asset_type_and_isin(test_db: DatabaseManager):
    # Given: DB with mixed stock and ETF records

    # When: Filtering for asset_type='etf'
    etfs = test_db.filter_financial_records(asset_type="etf")

    # Then: Only VUAA ETF record is returned
    assert len(etfs) == 1
    assert isinstance(etfs[0], TradeRecord)
    assert etfs[0].symbol == "VUAA"

    # When: Filtering for ISIN 'US0378331005'
    appls = test_db.filter_financial_records(isin="US0378331005")

    # Then: Only Apple stock record is returned
    assert len(appls) == 1
    assert isinstance(appls[0], TradeRecord)
    assert appls[0].symbol == "AAPL"


def test_filter_by_quantity_bounds(test_db: DatabaseManager):
    # Given: DB with records of quantities 50, 10, 5

    # When: Filtering quantity_over=Decimal("10") (>= 10)
    large_q: list[BaseStrictRecord] = test_db.filter_financial_records(quantity_over=Decimal("10"))

    # Then: AAPL (50) and VUAA (10) are returned
    assert len(large_q) == 2
    assert isinstance(large_q[0], TradeRecord)
    assert isinstance(large_q[1], TradeRecord)
    large_q_casted: list[TradeRecord] = large_q  # pyright: ignore[reportAssignmentType]
    symbols = [r.symbol for r in large_q_casted]
    assert "AAPL" in symbols
    assert "VUAA" in symbols

    # When: Filtering quantity_less=Decimal("8") (<= 8)
    small_q = test_db.filter_financial_records(quantity_less=Decimal("8"))

    # Then: Only MSFT (5) is returned
    assert len(small_q) == 1
    assert isinstance(small_q[0], TradeRecord)
    assert small_q[0].symbol == "MSFT"


def test_filter_by_purchase_date_range(test_db: DatabaseManager):
    # Given: DB with records in Jan, Mar, Jun 2024

    # When: Filtering date range between Feb 2024 and Apr 2024
    mid_year = test_db.filter_financial_records(
        purchase_date_start=datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc),
        purchase_date_end=datetime(2024, 4, 30, 23, 59, 59, tzinfo=timezone.utc),
    )

    # Then: Only March transaction (VUAA) is returned
    assert len(mid_year) == 1
    assert isinstance(mid_year[0], TradeRecord)
    assert mid_year[0].symbol == "VUAA"


def test_filter_logic_and_and_or(test_db: DatabaseManager):
    # Given: DB with multiple records

    # When: Filtering asset_type='stock' OR isin='IE00BFMXXD85' with logic='OR'
    or_results = test_db.filter_financial_records(
        asset_type="stock",
        isin="IE00BFMXXD85",
        logic="OR",
    )

    # Then: All 3 records match (2 stocks + 1 ETF matching ISIN)
    assert len(or_results) == 3

    # When: Filtering asset_type='stock' AND isin='IE00BFMXXD85' with logic='AND'
    and_results = test_db.filter_financial_records(
        asset_type="stock",
        isin="IE00BFMXXD85",
        logic="AND",
    )

    # Then: No record matches both criteria
    assert len(and_results) == 0
