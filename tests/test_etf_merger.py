"""Unit tests for AssetMerger model, DatabaseManager merger queries, and FIFO process_merger engine."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import AssetMerger
from backend.services.accounting.fifo import FIFOAccounting


@pytest.fixture
def db_manager():
    mgr = DatabaseManager(MemoryDb())
    yield mgr
    mgr.close()


def test_asset_merger_db_crud(db_manager: DatabaseManager):
    # Given: An ETF merger record
    merger = AssetMerger(
        old_isin="LU1781541179",
        new_isin="IE000BI8OT95",
        old_symbol="LCWD",
        new_symbol="MWRD",
        effective_date=datetime(2025, 2, 21, 0, 0, tzinfo=timezone.utc),
        exchange_ratio=Decimal("1.0"),
    )

    # When: Inserting asset merger into database
    inserted = db_manager.insert_asset_merger(merger)

    # Then: Merger record is assigned ID and queried correctly
    assert inserted.id is not None
    assert inserted.old_isin == "LU1781541179"
    assert inserted.new_isin == "IE000BI8OT95"

    all_mergers = db_manager.get_asset_mergers()
    assert len(all_mergers) == 1
    assert all_mergers[0].old_symbol == "LCWD"

    found = db_manager.get_merger_by_old_isin("LU1781541179")
    assert found is not None
    assert found.new_isin == "IE000BI8OT95"

    # When: Deleting asset merger
    deleted = db_manager.delete_asset_merger(inserted.id)
    assert deleted is True
    assert len(db_manager.get_asset_mergers()) == 0


def test_fifo_process_merger_with_fractional_share():
    # Given: FIFO engine with purchase lots for old asset LU1781541179
    fifo = FIFOAccounting()
    fifo.add_purchase(
        asset="LU1781541179",
        acquisition_date=datetime.fromisoformat("2024-01-15"),
        quantity=Decimal("10"),
        unit_price=Decimal("15.00"),
        fees=Decimal("1.00"),
    )
    fifo.add_purchase(
        asset="LU1781541179",
        acquisition_date=datetime.fromisoformat("2024-06-20"),
        quantity=Decimal("5.25"),
        unit_price=Decimal("20.00"),
        fees=Decimal("0.50"),
    )

    # Total old quantity: 15.25 shares
    expected_lot_count = 2
    assert len(fifo.get_holdings("LU1781541179")) == expected_lot_count

    # When: Processing corporate merger into new asset IE000BI8OT95 with exchange_ratio 1.0
    result = fifo.process_merger(
        old_asset="LU1781541179",
        new_asset="IE000BI8OT95",
        merger_date=datetime.fromisoformat("2025-02-21"),
        exchange_ratio=Decimal("1.0"),
    )

    # Then: Old asset lots are cleared, transferred to new asset, and fractional quantity calculated
    assert len(fifo.get_holdings("LU1781541179")) == 0
    new_holdings = fifo.get_holdings("IE000BI8OT95")
    assert len(new_holdings) == expected_lot_count

    assert result.total_old_quantity == Decimal("15.25")
    assert result.total_new_quantity == Decimal("15.25")
    assert result.whole_quantity == Decimal("15")
    assert result.fractional_quantity == Decimal("0.25")

    # Historical purchase dates and cost basis preserved
    assert new_holdings[0].acquisition_date == datetime.fromisoformat("2024-01-15")
    assert new_holdings[1].acquisition_date == datetime.fromisoformat("2024-06-20")


def test_fifo_process_merger_zero_ratio():
    fifo = FIFOAccounting()
    fifo.add_purchase(
        asset="OLD_ISIN",
        acquisition_date=datetime.fromisoformat("2024-01-15"),
        quantity=Decimal("10"),
        unit_price=Decimal("15.00"),
        fees=Decimal("1.00"),
    )
    with pytest.raises(ValueError, match="exchange_ratio must be positive"):
        fifo.process_merger(
            old_asset="OLD_ISIN",
            new_asset="NEW_ISIN",
            merger_date=datetime.fromisoformat("2025-02-21"),
            exchange_ratio=Decimal("0"),
        )


def test_fifo_process_merger_no_holdings():
    fifo = FIFOAccounting()
    result = fifo.process_merger(
        old_asset="OLD_ISIN",
        new_asset="NEW_ISIN",
        merger_date=datetime.fromisoformat("2025-02-21"),
        exchange_ratio=Decimal("1.0"),
    )
    assert result.total_old_quantity == Decimal("0")
    assert result.transformed_lots == 0


def test_fifo_process_merger_fractional_ratio():
    fifo = FIFOAccounting()
    fifo.add_purchase(
        asset="OLD_ISIN",
        acquisition_date=datetime.fromisoformat("2024-01-15"),
        quantity=Decimal("10"),
        unit_price=Decimal("15.00"),
        fees=Decimal("1.00"),
    )
    fifo.process_merger(
        old_asset="OLD_ISIN",
        new_asset="NEW_ISIN",
        merger_date=datetime.fromisoformat("2025-02-21"),
        exchange_ratio=Decimal("0.5"),
    )
    holdings = fifo.get_holdings("NEW_ISIN")
    assert len(holdings) == 1
    assert holdings[0].remaining_quantity == Decimal("5.0")
    assert holdings[0].unit_price == Decimal("30.00")


def test_fifo_process_merger_multiple():
    fifo = FIFOAccounting()
    fifo.add_purchase(
        asset="OLD_ISIN",
        acquisition_date=datetime.fromisoformat("2024-01-15"),
        quantity=Decimal("10"),
        unit_price=Decimal("15.00"),
        fees=Decimal("1.00"),
    )
    fifo.process_merger(
        old_asset="OLD_ISIN",
        new_asset="MID_ISIN",
        merger_date=datetime.fromisoformat("2024-02-21"),
        exchange_ratio=Decimal("1.0"),
    )
    fifo.process_merger(
        old_asset="MID_ISIN",
        new_asset="NEW_ISIN",
        merger_date=datetime.fromisoformat("2025-02-21"),
        exchange_ratio=Decimal("0.5"),
    )
    holdings = fifo.get_holdings("NEW_ISIN")
    assert len(holdings) == 1
    assert holdings[0].remaining_quantity == Decimal("5.0")
    assert holdings[0].unit_price == Decimal("30.00")
