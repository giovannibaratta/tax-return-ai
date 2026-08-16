"""Automated ETF NAV and Merger Ratio Resolver module.

Fetches historical NAVs directly from Euronext API (with fallback to yfinance),
resolves exchange ratios, and computes expected whole/fractional share breakdowns.
"""

import contextlib
import io
import json
import logging
import urllib.request
from datetime import datetime, timedelta
from decimal import Decimal

import yfinance as yf
from pydantic import BaseModel, ConfigDict

from backend.ingestion.openfigi import OpenFIGIMapper

logger = logging.getLogger(__name__)


class EuronextChartEntry(BaseModel):
    """Single price data point from Euronext chart API response."""

    time: str
    price: float | int | str


class MergerDetails(BaseModel):
    """Resolution details for an ETF merger calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    old_isin: str
    new_isin: str
    old_symbol: str
    new_symbol: str
    old_name: str | None = None
    new_name: str | None = None
    valuation_date: datetime
    nav_old: Decimal | None = None
    nav_new: Decimal | None = None
    exchange_ratio: Decimal | None = None
    old_quantity: Decimal
    total_new_shares: Decimal
    whole_shares: Decimal
    fractional_shares: Decimal
    expected_cash_payout: Decimal


class NAVResolver:
    """Automated historical NAV retriever via Euronext API for ETF mergers."""

    # Market Identifier Codes (MICs) for Euronext exchanges:
    # XPAR: Euronext Paris, XAMS: Euronext Amsterdam, XBRU: Euronext Brussels,
    # XLIS: Euronext Lisbon, XMSM: Euronext Milan (Euronext Growth Milan / Borsa Italiana)
    EURONEXT_MICS: list[str] = ["XPAR", "XAMS", "XBRU", "XLIS", "XMSM"]

    # User-Agent header simulates a standard web browser to prevent Euronext API blocking (HTTP 403 Forbidden).
    HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }

    def __init__(self) -> None:
        """Initialize NAVResolver with OpenFIGIMapper dependency."""
        self.figi_mapper = OpenFIGIMapper()

    def _fetch_euronext_nav(self, isin: str, target_date: datetime) -> Decimal | None:
        """Fetch historical daily NAV price for an ISIN from Euronext API on target_date.

        Args:
            isin: Asset ISIN identifier.
            target_date: Target valuation timestamp.

        Returns:
            Decimal price if exact date match found, None otherwise.
        """
        target_str = target_date.strftime("%Y-%m-%d")

        for mic in self.EURONEXT_MICS:
            url = f"https://live.euronext.com/en/ajax/getChartData/{isin}-{mic}/max"
            try:
                req = urllib.request.Request(url, headers=self.HEADERS)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    raw_data: object = json.loads(resp.read().decode("utf-8"))
                    if isinstance(raw_data, list):
                        for raw_item in raw_data:  # pyright: ignore[reportUnknownVariableType]
                            if isinstance(raw_item, dict):
                                try:
                                    entry = EuronextChartEntry.model_validate(raw_item)
                                    if target_str in entry.time:
                                        return Decimal(str(entry.price))
                                except Exception:
                                    continue
            except Exception as exc:
                logger.debug(f"Euronext lookup failed for {isin}-{mic}: {exc}")
                continue

        return None

    def _fetch_yfinance_nav(self, ticker_symbol: str, target_date: datetime) -> Decimal | None:
        """Fallback yfinance lookup for non-Euronext listed assets on exact target date.

        Args:
            ticker_symbol: Base ticker symbol of the security.
            target_date: Target valuation timestamp.

        Returns:
            Decimal price if exact target date close price exists, None otherwise.
        """
        target_str = target_date.strftime("%Y-%m-%d")
        start_date = target_str
        end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
        candidate_tickers = [f"{ticker_symbol}{suffix}" for suffix in [".MI", ".PA", ".DE", ".F", ".L", ""]]

        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            for cand in candidate_tickers:
                try:
                    t = yf.Ticker(cand)
                    hist = t.history(start=start_date, end=end_date)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    if hist is not None and not hist.empty and "Close" in hist.columns:  # pyright: ignore[reportUnknownMemberType]
                        for index_val, row in hist.iterrows():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                            idx_str = str(index_val)[:10]
                            if idx_str == target_str:
                                close_val: object = row["Close"]  # pyright: ignore[reportUnknownVariableType]
                                if isinstance(close_val, (int, float, str, Decimal)):
                                    val_float = float(close_val)
                                    if val_float > 0:
                                        return Decimal(str(round(val_float, 4)))
                except Exception:
                    continue

        return None

    def _resolve_single_nav(
        self,
        isin: str,
        ticker_symbol: str,
        valuation_date: datetime,
        nav_override: Decimal | None,
    ) -> Decimal | None:
        """Resolve NAV for a single asset using override, Euronext API, or yfinance fallback.

        Args:
            isin: Asset ISIN identifier.
            ticker_symbol: Ticker symbol for fallback resolution.
            valuation_date: Target valuation timestamp.
            nav_override: Optional manually supplied NAV price.

        Returns:
            Resolved Decimal NAV or None if unavailable.
        """
        if nav_override is not None:
            return nav_override
        euronext_nav = self._fetch_euronext_nav(isin, valuation_date)
        if euronext_nav is not None:
            return euronext_nav
        return self._fetch_yfinance_nav(ticker_symbol, valuation_date)

    def resolve_merger_details(
        self,
        old_isin: str,
        new_isin: str,
        valuation_date: datetime,
        old_quantity: Decimal | None = None,
        old_symbol_override: str | None = None,
        new_symbol_override: str | None = None,
        old_nav_override: Decimal | None = None,
        new_nav_override: Decimal | None = None,
    ) -> MergerDetails:
        """Resolve tickers, fetch historical NAVs, and compute merger ratio & fractional breakdown.

        Args:
            old_isin: ISIN identifier of target (old) fund.
            new_isin: ISIN identifier of acquiring (new) fund.
            valuation_date: Date of exchange calculation.
            old_quantity: Optional quantity of old shares held.
            old_symbol_override: Optional user override ticker for old fund.
            new_symbol_override: Optional user override ticker for new fund.
            old_nav_override: Optional user override NAV for old fund.
            new_nav_override: Optional user override NAV for new fund.

        Returns:
            MergerDetails object containing resolved tickers, NAVs, ratio, and share breakdown.

        Raises:
            ValueError: If ticker symbol cannot be resolved for either ISIN.
        """
        old_res = self.figi_mapper.map_isin(old_isin)
        old_ticker, old_name = old_res.ticker, old_res.name
        new_res = self.figi_mapper.map_isin(new_isin)
        new_ticker, new_name = new_res.ticker, new_res.name

        old_sym = old_symbol_override or old_ticker
        if not old_sym:
            raise ValueError(
                f"Could not resolve ticker symbol for old ISIN '{old_isin}'. Please provide a symbol override."
            )

        new_sym = new_symbol_override or new_ticker
        if not new_sym:
            raise ValueError(
                f"Could not resolve ticker symbol for new ISIN '{new_isin}'. Please provide a symbol override."
            )

        nav_old = self._resolve_single_nav(old_isin, old_sym, valuation_date, old_nav_override)
        nav_new = self._resolve_single_nav(new_isin, new_sym, valuation_date, new_nav_override)

        exchange_ratio = None
        if nav_old is not None and nav_new is not None and nav_new > Decimal("0"):
            exchange_ratio = (nav_old / nav_new).quantize(Decimal("0.000001"))

        # 3. Calculate share breakdown if quantity available
        total_old_qty = old_quantity or Decimal("0")
        total_new_qty = (total_old_qty * exchange_ratio) if exchange_ratio else Decimal("0")
        whole_shares = Decimal(int(total_new_qty))
        fractional_shares = total_new_qty - whole_shares

        expected_cash_payout = (fractional_shares * nav_new) if (nav_new and exchange_ratio) else Decimal("0")

        return MergerDetails(
            old_isin=old_isin,
            new_isin=new_isin,
            old_symbol=old_sym,
            new_symbol=new_sym,
            old_name=old_name,
            new_name=new_name,
            valuation_date=valuation_date,
            nav_old=nav_old,
            nav_new=nav_new,
            exchange_ratio=exchange_ratio,
            old_quantity=total_old_qty,
            total_new_shares=total_new_qty,
            whole_shares=whole_shares,
            fractional_shares=fractional_shares,
            expected_cash_payout=expected_cash_payout.quantize(Decimal("0.01")),
        )
