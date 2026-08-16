"""Unit tests for portfolio reconstruction engine and models."""

from datetime import datetime, timezone
from decimal import Decimal

from backend.db_models import AssetMerger
from backend.domain_models import BaseStrictRecord
from backend.portfolio import PortfolioEngine, PortfolioFilter, PortfolioSnapshot
from tests.factories import build_mock_record


def test_reconstruct_portfolio_basic_buy_and_hold():
    # Given: Single buy transaction for AAPL
    rec = build_mock_record(
        id=1,
        provider="interactive_brokers",
        account_country="ireland",
        event_timestamp=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("150.00"),
        total_amount=Decimal("1500.00"),
    )

    # When: Reconstructing portfolio
    snapshot = PortfolioEngine.reconstruct_portfolio([rec])

    # Then: Position should show active quantity, correct cost basis, zero realized P&L
    assert isinstance(snapshot, PortfolioSnapshot)
    assert snapshot.total_active_positions == 1
    assert snapshot.total_cost_basis == Decimal("1500.00")
    assert snapshot.total_realized_pnl == Decimal("0")
    assert snapshot.total_dividends == Decimal("0")
    assert len(snapshot.positions) == 1

    pos = snapshot.positions[0]
    assert pos.symbol == "AAPL"
    assert pos.current_quantity == Decimal("10")
    assert pos.average_buy_price == Decimal("150.00")
    assert pos.cost_basis == Decimal("1500.00")


def test_reconstruct_portfolio_partial_sell_and_pnl():
    # Given: Two buys and one partial sell for AAPL
    rec1 = build_mock_record(
        id=1,
        event_timestamp=datetime(2025, 1, 10, tzinfo=timezone.utc),
        symbol="AAPL",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100.00"),
        total_amount=Decimal("1000.00"),
    )

    rec2 = build_mock_record(
        id=2,
        event_timestamp=datetime(2025, 2, 10, tzinfo=timezone.utc),
        symbol="AAPL",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("200.00"),
        total_amount=Decimal("2000.00"),
    )

    rec3 = build_mock_record(
        id=3,
        event_timestamp=datetime(2025, 3, 10, tzinfo=timezone.utc),
        symbol="AAPL",
        action="sell",
        quantity=Decimal("10"),
        unit_price=Decimal("250.00"),
        total_amount=Decimal("2500.00"),
    )

    # When: Reconstructing portfolio
    snapshot = PortfolioEngine.reconstruct_portfolio([rec1, rec2, rec3])

    # Then: Realized P&L calculated correctly and current holdings updated
    # FIFO Lot Matching:
    # Buy 1: 10 shares @ €100 = €1000 cost basis.
    # Buy 2: 10 shares @ €200 = €2000 cost basis.
    # Sold: 10 shares @ €250 = €2500 proceeds -> FIFO consumes Buy 1 (cost = €1000).
    # Realized P&L = €2500 - €1000 = €1500.
    # Remaining: 10 shares from Buy 2 -> cost basis = €2000, avg unit price = €200.

    assert snapshot.total_active_positions == 1
    assert snapshot.total_cost_basis == Decimal("2000.00")
    assert snapshot.total_realized_pnl == Decimal("1500.00")

    pos = snapshot.positions[0]
    assert pos.current_quantity == Decimal("10")
    assert pos.average_buy_price == Decimal("200.00")
    assert pos.cost_basis == Decimal("2000.00")
    assert pos.realized_pnl == Decimal("1500.00")


def test_reconstruct_portfolio_dividends_and_closed_position():
    # Given: Full buy, sell, and dividend payout for ETF
    rec_buy = build_mock_record(
        id=10,
        event_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        action="buy",
        quantity=Decimal("5"),
        unit_price=Decimal("80.00"),
        total_amount=Decimal("400.00"),
    )
    rec_div = build_mock_record(
        id=11,
        event_timestamp=datetime(2025, 2, 1, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        action="dividend",
        quantity=Decimal("0"),
        unit_price=Decimal("0"),
        total_amount=Decimal("25.00"),
    )
    rec_sell = build_mock_record(
        id=12,
        event_timestamp=datetime(2025, 3, 1, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        action="sell",
        quantity=Decimal("5"),
        unit_price=Decimal("90.00"),
        total_amount=Decimal("450.00"),
    )

    records = [rec_buy, rec_div, rec_sell]

    # When: Querying with default 'active' filter vs 'all' or 'closed' status
    # Active filter (default) -> position closed so 0 positions returned
    active_snapshot = PortfolioEngine.reconstruct_portfolio(records, PortfolioFilter(position_status="active"))

    # Then: Filters accurately toggle position visibility
    assert len(active_snapshot.positions) == 0

    # Closed filter -> position returned
    closed_snapshot = PortfolioEngine.reconstruct_portfolio(records, PortfolioFilter(position_status="closed"))
    assert len(closed_snapshot.positions) == 1
    pos = closed_snapshot.positions[0]
    assert pos.symbol == "VUAA"
    assert pos.current_quantity == Decimal("0")
    assert pos.realized_pnl == Decimal("50.00")
    assert pos.total_dividends == Decimal("25.00")


def test_reconstruct_portfolio_filtering():
    # Given: Multiple positions across jurisdictions and providers
    rec_it = build_mock_record(
        id=1,
        provider="directa",
        account_country="italy",
        event_timestamp=datetime(2025, 1, 15, tzinfo=timezone.utc),
        symbol="ENI",
        action="buy",
        quantity=Decimal("100"),
        unit_price=Decimal("14.0"),
        total_amount=Decimal("1400"),
    )
    rec_ie = build_mock_record(
        id=2,
        provider="interactive_brokers",
        account_country="ireland",
        asset_type="etf",
        event_timestamp=datetime(2025, 1, 15, tzinfo=timezone.utc),
        symbol="CSPX",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("500.0"),
        total_amount=Decimal("5000"),
    )

    records = [rec_it, rec_ie]

    # When: Applying jurisdiction, search query, and asset type filters
    # Jurisdiction filter
    it_snapshot = PortfolioEngine.reconstruct_portfolio(records, PortfolioFilter(account_country="italy"))

    # Then: Only matching positions are included in snapshot
    assert len(it_snapshot.positions) == 1
    assert it_snapshot.positions[0].symbol == "ENI"

    # Search query filter
    search_snapshot = PortfolioEngine.reconstruct_portfolio(records, PortfolioFilter(search_query="CSPX"))
    assert len(search_snapshot.positions) == 1
    assert search_snapshot.positions[0].symbol == "CSPX"


def test_portfolio_performance_measurement():
    # Given: 500 generated financial records
    records: list[BaseStrictRecord] = []
    for i in range(500):
        records.append(
            build_mock_record(
                id=i + 1,
                provider="directa" if i % 2 == 0 else "interactive_brokers",
                account_country="italy" if i % 2 == 0 else "ireland",
                event_timestamp=datetime(2025, 1, (i % 28) + 1, tzinfo=timezone.utc),
                asset_type="stock" if i % 3 == 0 else "etf",
                symbol=f"STOCK_{i % 20}",
                action="buy" if i % 5 != 0 else "sell",
                quantity=Decimal("10"),
                unit_price=Decimal("100.00"),
                total_amount=Decimal("1000.00"),
            )
        )

    # When: Reconstructing portfolio snapshot
    snapshot = PortfolioEngine.reconstruct_portfolio(records)

    # Then: Execution completes in < 50ms and elapsed_ms is measured
    assert snapshot.elapsed_ms < 50.0  # Well under 2000ms target
    assert len(snapshot.positions) > 0


def test_reconstruct_portfolio_with_mergers():
    # Given: Buy 10 shares of old_isin
    rec = build_mock_record(
        id=1,
        event_timestamp=datetime(2024, 1, 10, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="LCWD",
        isin="LU1781541179",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100.00"),
        total_amount=Decimal("1000.00"),
    )

    # Given: Sell 2 shares of old_isin BEFORE merger
    rec_sell_before = build_mock_record(
        id=2,
        event_timestamp=datetime(2024, 6, 15, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="LCWD",
        isin="LU1781541179",
        action="sell",
        quantity=Decimal("2"),
        unit_price=Decimal("110.00"),
        total_amount=Decimal("220.00"),
    )

    # Given: Merger mapping old ISIN to new ISIN
    merger = AssetMerger(
        old_isin="LU1781541179",
        new_isin="IE000BI8OT95",
        old_symbol="LCWD",
        new_symbol="MWRD",
        effective_date=datetime(2025, 2, 20, 0, 0, tzinfo=timezone.utc),
        exchange_ratio=Decimal("1.0"),
    )

    # Given: Sell 3 shares of new_isin AFTER merger
    rec_sell_after = build_mock_record(
        id=3,
        event_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="MWRD",
        isin="IE000BI8OT95",
        action="sell",
        quantity=Decimal("3"),
        unit_price=Decimal("120.00"),
        total_amount=Decimal("360.00"),
    )

    # When: Reconstructing portfolio with mergers
    snapshot = PortfolioEngine.reconstruct_portfolio(records=[rec, rec_sell_before, rec_sell_after], mergers=[merger])

    # Then: Position isin is updated and cost basis preserved
    assert len(snapshot.positions) == 1
    pos = snapshot.positions[0]
    assert pos.isin == "IE000BI8OT95"
    assert pos.symbol == "MWRD"

    assert pos.current_quantity == Decimal("5")
    assert pos.cost_basis == Decimal("500.00")
    assert snapshot.total_realized_pnl == Decimal("80.00")


def test_merger_mid_year_incoherence():
    # Given: Buy old ISIN before merger and again AFTER merger date
    rec_buy_valid = build_mock_record(
        id=1,
        event_timestamp=datetime(2024, 1, 10, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="LCWD",
        isin="LU1781541179",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        total_amount=Decimal("1000"),
    )
    rec_buy_invalid = build_mock_record(
        id=2,
        event_timestamp=datetime(2024, 6, 10, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="LCWD",
        isin="LU1781541179",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        total_amount=Decimal("1000"),
    )

    merger = AssetMerger(
        old_isin="LU1781541179",
        new_isin="IE000BI8OT95",
        old_symbol="LCWD",
        new_symbol="MWRD",
        effective_date=datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc),
        exchange_ratio=Decimal("1.0"),
    )

    # When: Reconstructing portfolio
    snapshot = PortfolioEngine.reconstruct_portfolio(records=[rec_buy_valid, rec_buy_invalid], mergers=[merger])

    # Then: Transactions on old ISIN after merger date flag as incoherent
    assert len(snapshot.positions) == 2

    # We should have one position for the new ISIN (from the first buy)
    pos_new = next(p for p in snapshot.positions if p.isin == "IE000BI8OT95")
    assert pos_new.current_quantity == Decimal("10")

    # And one position for the old ISIN (from the invalid second buy)
    pos_old = next(p for p in snapshot.positions if p.isin == "LU1781541179")
    assert pos_old.current_quantity == Decimal("10")
    assert pos_old.is_incoherent is True
    assert any("post merger effective date" in r for r in pos_old.incoherence_reasons)
