"""Unit tests for automated NAVResolver module."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.ingestion.nav_resolver import NAVResolver


def test_nav_resolver_calculations():
    # Given: NAVResolver instance
    resolver = NAVResolver()

    # Mock historical NAV fetching with synthetic values for testing
    def mock_euronext(isin: str, target_date: datetime) -> Decimal | None:
        if "IE000BI8OT95" in isin:
            return Decimal("136.5200")
        if "LU1781541179" in isin:
            return Decimal("19.8100")
        return None

    resolver._fetch_euronext_nav = mock_euronext

    # When: Resolving merger details for 1896 old shares
    res = resolver.resolve_merger_details(
        old_isin="LU1781541179",
        new_isin="IE000BI8OT95",
        valuation_date=datetime(2025, 2, 20, 0, 0, tzinfo=timezone.utc),
        old_quantity=Decimal("1896"),
    )

    # Then: Tickers resolved, NAVs fetched, exchange ratio calculated
    assert res.old_symbol is not None
    assert res.new_symbol is not None
    assert res.nav_old == Decimal("19.8100")
    assert res.nav_new == Decimal("136.5200")

    # Ratio: 19.81 / 136.52 = 0.145107
    expected_ratio = (Decimal("19.8100") / Decimal("136.5200")).quantize(Decimal("0.000001"))
    assert res.exchange_ratio == expected_ratio

    # Whole shares: int(1896 * 0.145107) = 275
    assert res.whole_shares == Decimal("275")
    # Fractional payout calculation is non-zero
    assert res.fractional_shares > Decimal("0")
    assert res.expected_cash_payout > Decimal("0")


def test_nav_resolver_missing_ticker_raises_value_error() -> None:
    from backend.ingestion.openfigi import FIGIMappingResult

    resolver = NAVResolver()
    resolver.figi_mapper.map_isin = lambda isin: FIGIMappingResult(ticker=None, name=None)

    # When: Resolving merger details without ticker overrides
    # Then: ValueError is raised requiring symbol override
    with pytest.raises(ValueError, match="Could not resolve ticker symbol"):
        resolver.resolve_merger_details(
            old_isin="UNKNOWN_ISIN_1",
            new_isin="UNKNOWN_ISIN_2",
            valuation_date=datetime(2025, 2, 20, 0, 0, tzinfo=timezone.utc),
        )
