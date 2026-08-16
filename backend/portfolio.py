"""Portfolio aggregation engine for reconstructing asset holdings from raw financial records using FIFO lot tracking."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

if TYPE_CHECKING:
    from backend.db_manager import DatabaseManager

from backend.db_models import AssetMerger
from backend.domain_models import (
    AssetIdentity,
    BaseStrictRecord,
    DividendRecord,
    SecurityRecord,
    TradeRecord,
    TransactionAction,
)
from backend.services.accounting.fifo import FIFOAccounting


@dataclass
class MergerTransformationResult:
    """Result of processing a corporate action/merger on a financial transaction record."""

    transformed_id: str
    isin: str
    symbol: str
    name: str
    adjusted_quantity: Decimal
    inconsistency_reason: str | None


class PortfolioPosition(BaseModel):
    """Aggregated portfolio position snapshot for a specific asset computed using FIFO lot tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str

    identity: AssetIdentity
    asset_type: str
    provider: str
    account_country: str
    currency: str
    accounting_method: str = Field(
        default="FIFO",
        description=(
            "Accounting methodology used for lot tracking and cost basis calculations. "
            "Set to 'FIFO' (First-In, First-Out) chronological lot matching."
        ),
    )

    _total_bought_qty: Decimal = PrivateAttr(default=Decimal("0"))
    _total_sold_qty: Decimal = PrivateAttr(default=Decimal("0"))
    _total_bought_cost: Decimal = PrivateAttr(default=Decimal("0"))
    _total_sold_proceeds: Decimal = PrivateAttr(default=Decimal("0"))
    _total_dividends: Decimal = PrivateAttr(default=Decimal("0"))
    _accumulated_fifo_realized_pnl: Decimal = PrivateAttr(default=Decimal("0"))

    _last_activity_date: datetime | None = PrivateAttr(default=None)
    _transaction_count: int = PrivateAttr(default=0)

    _is_incoherent: bool = PrivateAttr(default=False)
    _incoherence_reasons: list[str] = PrivateAttr(default_factory=list)

    _fifo_engine: FIFOAccounting = PrivateAttr(default_factory=FIFOAccounting)

    @computed_field
    @property
    def symbol(self) -> str | None:
        """Trading symbol of the asset."""
        return self.identity.symbol

    @computed_field
    @property
    def isin(self) -> str | None:
        """ISIN code of the asset."""
        return self.identity.isin

    @computed_field
    @property
    def asset_name(self) -> str | None:
        """Human-readable asset description or security name."""
        return self.identity.asset_name

    @computed_field
    @property
    def total_bought_qty(self) -> Decimal:
        """Cumulative quantity bought across all transactions."""
        return self._total_bought_qty

    @computed_field
    @property
    def total_sold_qty(self) -> Decimal:
        """Cumulative quantity sold across all transactions."""
        return self._total_sold_qty

    @computed_field
    @property
    def total_bought_cost(self) -> Decimal:
        """Cumulative capital invested across all buy transactions."""
        return self._total_bought_cost

    @computed_field
    @property
    def total_sold_proceeds(self) -> Decimal:
        """Cumulative gross proceeds received across all sell transactions."""
        return self._total_sold_proceeds

    @computed_field
    @property
    def total_dividends(self) -> Decimal:
        """Cumulative dividends received for this asset position."""
        return self._total_dividends

    @computed_field
    @property
    def last_activity_date(self) -> datetime | None:
        """Timestamp of the most recent transaction recorded for this position."""
        return self._last_activity_date

    @computed_field
    @property
    def transaction_count(self) -> int:
        """Total number of transactions processed for this position."""
        return self._transaction_count

    @computed_field
    @property
    def is_incoherent(self) -> bool:
        """Flag indicating whether position contains an incoherent transaction sequence."""
        return self._is_incoherent

    @computed_field
    @property
    def incoherence_reasons(self) -> list[str]:
        """List of detected incoherence reasons."""
        return list(self._incoherence_reasons)

    @computed_field
    @property
    def current_quantity(self) -> Decimal:
        """Net current holdings quantity remaining across open FIFO purchase lots."""
        holdings = self._fifo_engine.get_holdings(self.key)
        if not holdings:
            return Decimal("0")
        raw_net = sum((h.remaining_quantity for h in holdings), Decimal("0"))
        if raw_net > Decimal("0") and raw_net == raw_net.to_integral_value():
            return Decimal(int(raw_net))
        return raw_net

    @computed_field
    @property
    def cost_basis(self) -> Decimal:
        """Total cost basis of remaining open FIFO purchase lots (including allocated buy fees)."""
        holdings = self._fifo_engine.get_holdings(self.key)
        if not holdings:
            return Decimal("0")
        return sum((h.cost_basis_remaining for h in holdings), Decimal("0"))

    @computed_field
    @property
    def average_buy_price(self) -> Decimal:
        """Effective unit cost basis of remaining open FIFO purchase lots (Cost Basis / Holdings Qty)."""
        qty = self.current_quantity
        if qty > Decimal("0"):
            return self.cost_basis / qty
        return Decimal("0")

    @computed_field
    @property
    def realized_pnl(self) -> Decimal:
        """Realized profit and loss calculated via chronological FIFO lot matching."""
        return self._accumulated_fifo_realized_pnl

    def _mark_incoherent(self, reason: str) -> None:
        """Flag position as incoherent and append reason if not present."""
        self._is_incoherent = True
        if reason not in self._incoherence_reasons:
            self._incoherence_reasons.append(reason)

    def _reevaluate_incoherence(self) -> None:
        """Re-evaluate state-based incoherence conditions after transaction additions."""
        if self._total_sold_qty > Decimal("0") and self._total_bought_qty == Decimal("0"):
            self._mark_incoherent("Sell transaction recorded with zero prior BUY records (orphan sell).")
        if self._total_bought_qty - self._total_sold_qty < Decimal("0"):
            self._mark_incoherent(
                f"Negative holding quantity detected ({self._total_bought_qty - self._total_sold_qty})."
            )
        if self._total_sold_qty > self._total_bought_qty and self._total_bought_qty > Decimal("0"):
            self._mark_incoherent("Total sold quantity exceeds total bought quantity (oversold position).")

    def add_trade(self, rec: TradeRecord, adjusted_qty: Decimal, warning: str | None = None) -> None:
        """Accumulate a TradeRecord into position FIFO lots and re-evaluate invariants.

        Args:
            rec: Evaluated TradeRecord entity.
            adjusted_qty: Quantity after applying corporate merger scaling.
            warning: Optional corporate action or merger anomaly warning message.
        """
        self._transaction_count += 1
        if self._last_activity_date is None or rec.event_timestamp > self._last_activity_date:
            self._last_activity_date = rec.event_timestamp

        if warning:
            self._mark_incoherent(warning)

        if rec.action == TransactionAction.BUY:
            self._total_bought_qty += adjusted_qty
            self._total_bought_cost += rec.local_total_amount
            self._fifo_engine.add_purchase(
                asset=self.key,
                acquisition_date=rec.event_timestamp,
                quantity=adjusted_qty,
                unit_price=rec.unit_price,
                fees=rec.fees,
            )
        elif rec.action == TransactionAction.SELL:
            self._total_sold_qty += adjusted_qty
            self._total_sold_proceeds += rec.local_total_amount
            try:
                matches = self._fifo_engine.process_sale(
                    asset=self.key,
                    disposal_date=rec.event_timestamp,
                    quantity=adjusted_qty,
                    unit_price=rec.unit_price,
                    fees=rec.fees,
                )
                for m in matches:
                    self._accumulated_fifo_realized_pnl += m.realized_gain
            except ValueError as err:
                self._mark_incoherent(str(err))

        self._reevaluate_incoherence()

    def add_dividend(self, rec: DividendRecord, warning: str | None = None) -> None:
        """Accumulate a DividendRecord into position totals and re-evaluate invariants.

        Args:
            rec: Evaluated DividendRecord entity.
            warning: Optional corporate action or merger anomaly warning message.
        """
        self._transaction_count += 1
        if self._last_activity_date is None or rec.event_timestamp > self._last_activity_date:
            self._last_activity_date = rec.event_timestamp

        if warning:
            self._mark_incoherent(warning)

        self._total_dividends += rec.local_total_amount
        self._reevaluate_incoherence()


class PortfolioFilter(BaseModel):
    """Filter criteria for portfolio snapshot query.

    Attributes:
        account_country: Account country filter (e.g. 'italy', 'ireland', or 'all').
        provider: Financial service provider / broker filter (e.g. 'directa', 'interactive_brokers').
        asset_type: Asset type filter (e.g. 'stock', 'etf').
        position_status: Filter by active positions (> 0 qty), closed positions (0 qty), or all positions.
        search_query: Free-text search matching symbol, ISIN, or asset name.
        as_of_date: Historical cutoff timestamp. Only financial records with an event date
            on or before this timestamp are included in the snapshot.
    """

    account_country: str | None = None
    provider: str | None = None
    asset_type: str | None = None
    position_status: Literal["active", "closed", "all"] = "active"
    search_query: str | None = None
    as_of_date: datetime | None = Field(
        default=None,
        description="Historical cutoff timestamp. Filters transactions occurring on or before this date.",
    )


class PortfolioSnapshot(BaseModel):
    """Complete portfolio snapshot as of a specific date with execution latency metrics."""

    snapshot_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accounting_method: str = Field(
        default="FIFO",
        description=(
            "Accounting methodology used for portfolio reconstruction and lot tracking. "
            "Set to 'FIFO' (First-In, First-Out) chronological lot matching."
        ),
    )
    elapsed_ms: float = 0.0
    total_active_positions: int = 0
    incoherent_positions_count: int = 0
    total_cost_basis: Decimal = Field(default=Decimal("0"))
    total_realized_pnl: Decimal = Field(default=Decimal("0"))
    total_dividends: Decimal = Field(default=Decimal("0"))
    positions: list[PortfolioPosition] = Field(default_factory=list[PortfolioPosition])


class PortfolioEngine:
    """Computes portfolio holdings on-the-fly from strict financial domain records using FIFO lot tracking."""

    @staticmethod
    def _filter_and_sort_records(
        records: list[BaseStrictRecord],
        cutoff_date: datetime | None,
    ) -> list[BaseStrictRecord]:
        """Filter records by optional historical cutoff date and sort chronologically."""
        active_records = records
        if cutoff_date:
            cutoff = cutoff_date
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            active_records = [r for r in records if r.event_timestamp <= cutoff]
        return sorted(active_records, key=lambda r: r.event_timestamp)

    @staticmethod
    def _process_merger_transformation(
        rec: SecurityRecord,
        rec_qty: Decimal,
        merger_map: dict[str, AssetMerger],
    ) -> MergerTransformationResult:
        """Apply ISIN remapping and exchange ratio scaling if record falls on or before effective merger date.

        Args:
            rec: SecurityRecord to inspect.
            rec_qty: Original transaction quantity.
            merger_map: Mapping of lowercased old ISIN to AssetMerger.

        Returns:
            MergerTransformationResult instance.
            `transformed_id` encapsulates the internal fallback logic (ISIN -> Symbol -> original ID) 
            so that caller can safely group pre-merger and post-merger transactions under a single unified key.
            If a transaction with an old ISIN occurs strictly after the merger effective date,
            inconsistency_reason returns a descriptive error message.
        """
        rec_isin = rec.isin or ""
        rec_symbol = rec.symbol or ""
        rec_name = rec.asset_name or ""
        inconsistency: str | None = None

        if rec_isin.lower() in merger_map:
            m = merger_map[rec_isin.lower()]
            m_eff = (
                m.effective_date
                if m.effective_date.tzinfo is not None
                else m.effective_date.replace(tzinfo=timezone.utc)
            )
            rec_dt = (
                rec.event_timestamp
                if rec.event_timestamp.tzinfo is not None
                else rec.event_timestamp.replace(tzinfo=timezone.utc)
            )

            if rec_dt > m_eff:
                inconsistency = (
                    f"Transaction on {rec_dt.strftime('%Y-%m-%d')} uses old ISIN '{rec_isin}' "
                    f"post merger effective date ({m_eff.strftime('%Y-%m-%d')})."
                )
            else:
                rec_isin = m.new_isin
                if m.new_symbol:
                    rec_symbol = m.new_symbol
                if rec_qty > Decimal("0") and m.exchange_ratio > Decimal("0"):
                    # Retain full Decimal precision during corporate exchange scaling
                    rec_qty = rec_qty * m.exchange_ratio

        transformed_id = rec_isin or rec_symbol or rec.asset_identifier
        return MergerTransformationResult(
            transformed_id=transformed_id,
            isin=rec_isin,
            symbol=rec_symbol,
            name=rec_name,
            adjusted_quantity=rec_qty,
            inconsistency_reason=inconsistency,
        )

    @classmethod
    def _accumulate_positions(
        cls,
        sorted_records: list[BaseStrictRecord],
        merger_map: dict[str, AssetMerger],
    ) -> dict[str, PortfolioPosition]:
        """Group records into composite position accumulators keyed by '{asset_identifier}|{provider}|{account_country}'.

        Args:
            sorted_records: Chronologically sorted list of strict domain records.
            merger_map: Map of lowercased old ISIN to AssetMerger configuration.

        Returns:
            Dictionary mapping position key string ('{asset_identifier}|{provider}|{account_country}')
            to the accumulated PortfolioPosition instance.
        """
        positions_map: dict[str, PortfolioPosition] = {}

        for rec in sorted_records:
            if not isinstance(rec, SecurityRecord):
                continue

            if isinstance(rec, TradeRecord):
                raw_qty = rec.quantity
            else:
                raw_qty = Decimal("0")

            result = cls._process_merger_transformation(rec, raw_qty, merger_map)

            asset_type_str = rec.asset_type.value
            prov = rec.provider
            acct_country = rec.account_country
            curr = rec.currency

            pos_key = f"{result.transformed_id}|{prov}|{acct_country}".lower()

            if pos_key not in positions_map:
                identity = AssetIdentity(
                    symbol=result.symbol,
                    isin=result.isin,
                    asset_name=result.name,
                )
                positions_map[pos_key] = PortfolioPosition(
                    key=pos_key,
                    identity=identity,
                    asset_type=asset_type_str,
                    provider=prov,
                    account_country=acct_country,
                    currency=curr,
                )

            pos = positions_map[pos_key]

            if isinstance(rec, TradeRecord):
                pos.add_trade(rec, result.adjusted_quantity, warning=result.inconsistency_reason)
            elif isinstance(rec, DividendRecord):
                pos.add_dividend(rec, warning=result.inconsistency_reason)

        return positions_map

    @staticmethod
    def _filter_position(
        pos: PortfolioPosition,
        f: PortfolioFilter,
    ) -> PortfolioPosition | None:
        """Apply PortfolioFilter criteria to a computed position."""
        if f.account_country and f.account_country.lower() != "all":
            if pos.account_country.lower() != f.account_country.lower():
                return None

        if f.provider and f.provider.lower() != "all":
            if (pos.provider or "").lower() != f.provider.lower():
                return None

        if f.asset_type and f.asset_type.lower() != "all":
            if pos.asset_type.lower() != f.asset_type.lower():
                return None

        if f.position_status == "active" and pos.current_quantity <= Decimal("0"):
            return None
        elif f.position_status == "closed" and pos.current_quantity > Decimal("0"):
            return None

        if f.search_query:
            q = f.search_query.strip().lower()
            sym_match = q in (pos.symbol or "").lower()
            isin_match = q in (pos.isin or "").lower()
            name_match = q in (pos.asset_name or "").lower()
            if not (sym_match or isin_match or name_match):
                return None

        return pos

    @classmethod
    def reconstruct_portfolio(
        cls,
        records: list[BaseStrictRecord],
        filters: PortfolioFilter | None = None,
        mergers: list[AssetMerger] | None = None,
    ) -> PortfolioSnapshot:
        """Reconstruct portfolio positions from a list of strict domain records using FIFO lot tracking.

        Corporate Merger Limitations:
            Corporate action transformation currently performs single-pass ISIN remapping (`merger_map[old_isin]`).
            Chained ISIN mergers (e.g. Asset A merges into B on Date 1, and B merges into C on Date 2) or multi-asset
            corporate splits are not automatically resolved multi-hop in a single pass. For chained events,
            entries must map intermediate old ISINs directly to target ISINs or be processed via multi-pass resolution.

        Accounting Methodology:
            Position quantities, cost bases, and realized P&L are computed using chronological
            First-In, First-Out (FIFO) purchase lot tracking.

        Args:
            records: List of validated BaseStrictRecord domain entities.
            filters: Optional filtering criteria to apply to the snapshot.
            mergers: Optional registered ETF mergers to remap old ISINs to new ISINs.

        Returns:
            PortfolioSnapshot containing aggregated FIFO position data and performance metrics.
        """
        start_time = time.perf_counter()
        f = filters or PortfolioFilter()
        merger_map = {m.old_isin.lower(): m for m in (mergers or [])}

        sorted_records = cls._filter_and_sort_records(records, f.as_of_date)
        positions_map = cls._accumulate_positions(sorted_records, merger_map)

        final_positions: list[PortfolioPosition] = []
        for pos in positions_map.values():
            filtered_pos = cls._filter_position(pos, f)
            if filtered_pos:
                final_positions.append(filtered_pos)

        total_active_pos = sum(1 for p in final_positions if p.current_quantity > Decimal("0"))
        total_incoherent_pos = sum(1 for p in final_positions if p.is_incoherent)
        total_cost = sum(
            (p.cost_basis for p in final_positions if p.current_quantity > Decimal("0")),
            Decimal("0"),
        )
        total_realized = sum((p.realized_pnl for p in final_positions), Decimal("0"))
        total_divs = sum((p.total_dividends for p in final_positions), Decimal("0"))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PortfolioSnapshot(
            snapshot_timestamp=datetime.now(timezone.utc),
            accounting_method="FIFO",
            elapsed_ms=round(elapsed_ms, 2),
            total_active_positions=total_active_pos,
            incoherent_positions_count=total_incoherent_pos,
            total_cost_basis=total_cost,
            total_realized_pnl=total_realized,
            total_dividends=total_divs,
            positions=final_positions,
        )

    @classmethod
    def get_snapshot(
        cls,
        db: DatabaseManager,
        filters: PortfolioFilter | None = None,
    ) -> PortfolioSnapshot:
        """Fetch records from DatabaseManager and compute portfolio snapshot.

        Args:
            db: DatabaseManager instance.
            filters: Optional PortfolioFilter object.

        Returns:
            PortfolioSnapshot containing aggregated position data.
        """
        records = db.filter_financial_records()
        mergers = db.get_asset_mergers()
        return cls.reconstruct_portfolio(records, filters, mergers)
