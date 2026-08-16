"""Chronological FIFO (First-In, First-Out) lot matching for capital gains tax."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from pydantic import BaseModel, ConfigDict, Field

from backend.domain_models import assert_postcondition


class FIFOHolding(BaseModel):
    """An open FIFO purchase lot with a positive unconsumed quantity.

    ``cost_basis_remaining`` is the exact pro-rata acquisition cost, including
    the acquisition fees that remain attached to this holding.
    """

    model_config = ConfigDict(frozen=True)

    acquisition_date: datetime = Field(description="Timestamp at which the source lot was acquired.")
    remaining_quantity: Decimal = Field(description="Positive quantity that has not yet been disposed of.")
    original_quantity: Decimal = Field(description="Quantity originally acquired in the source lot.")
    unit_price: Decimal = Field(description="Acquisition price per unit before acquisition fees.")
    acquisition_fees: Decimal = Field(description="Total acquisition fees originally charged for the lot.")
    cost_basis_remaining: Decimal = Field(description="Exact remaining acquisition cost including allocated fees.")


class FIFOMatch(BaseModel):
    """The portion of one purchase lot matched to a single disposal."""

    model_config = ConfigDict(frozen=True)

    asset: str = Field(description="Asset identifier shared by the matched purchase and disposal.")
    acquisition_date: datetime = Field(description="Timestamp of the matched purchase lot.")
    disposal_date: datetime = Field(description="Timestamp of the disposal.")
    quantity: Decimal = Field(description="Positive quantity matched from the purchase lot.")
    buy_unit_price: Decimal = Field(description="Purchase price per unit before acquisition fees.")
    sell_unit_price: Decimal = Field(description="Disposal price per unit before disposal fees.")
    buy_fees_pro_rata: Decimal = Field(description="Exact acquisition fees allocated to this match.")
    sell_fees_pro_rata: Decimal = Field(description="Exact disposal fees allocated to this match.")
    cost_basis: Decimal = Field(description="Exact acquisition cost allocated to this match.")
    sale_proceeds: Decimal = Field(description="Gross disposal proceeds before disposal fees.")
    net_proceeds: Decimal = Field(description="Disposal proceeds after allocated disposal fees.")
    realized_gain: Decimal = Field(description="Net disposal proceeds less matched cost basis.")


class FIFOMergerResult(BaseModel):
    """The aggregate result of a tax-neutral corporate-merger lot transformation."""

    model_config = ConfigDict(frozen=True)

    old_asset: str = Field(description="Asset identifier removed by the merger.")
    new_asset: str = Field(description="Replacement asset identifier created by the merger.")
    merger_date: datetime = Field(description="Effective timestamp of the corporate merger.")
    exchange_ratio: Decimal = Field(description="Replacement units issued for each old unit.")
    total_old_quantity: Decimal = Field(description="Total open old-asset quantity transformed.")
    total_new_quantity: Decimal = Field(description="Total replacement quantity issued, including fractions.")
    whole_quantity: Decimal = Field(description="Integral component of the replacement quantity.")
    fractional_quantity: Decimal = Field(description="Fractional component of the replacement quantity.")
    transformed_lots: int = Field(description="Number of open source lots preserved in the replacement asset.")


@dataclass(slots=True)
class _FIFOLot:
    """Internal mutable balances associated with immutable acquisition metadata."""

    metadata: _LotMetadata
    remaining_quantity: Decimal
    remaining_acquisition_fees: Decimal
    remaining_cost_basis: Decimal

    @property
    def asset(self) -> str:
        """Return the immutable asset identifier."""
        return self.metadata.asset

    @property
    def acquisition_date(self) -> datetime:
        """Return the immutable source acquisition timestamp."""
        return self.metadata.acquisition_date

    @property
    def original_quantity(self) -> Decimal:
        """Return the immutable quantity originally acquired."""
        return self.metadata.original_quantity

    @property
    def unit_price(self) -> Decimal:
        """Return the immutable acquisition price per unit before fees."""
        return self.metadata.unit_price

    @property
    def acquisition_fees(self) -> Decimal:
        """Return the immutable original total acquisition fees."""
        return self.metadata.acquisition_fees

    @property
    def total_cost_basis(self) -> Decimal:
        """Return the immutable original lot acquisition cost including fees."""
        return (self.original_quantity * self.unit_price) + self.acquisition_fees


@dataclass(frozen=True, slots=True)
class _LotMetadata:
    """Immutable attributes established when an acquisition lot is created."""

    asset: str
    acquisition_date: datetime
    original_quantity: Decimal
    unit_price: Decimal
    acquisition_fees: Decimal

@dataclass(frozen=True, slots=True)
class _SaleAllocation:
    """A planned, validated consumption of one internal lot."""

    lot: _FIFOLot
    quantity: Decimal
    acquisition_fees: Decimal
    cost_basis: Decimal


class FIFOAccounting:
    """Maintain isolated asset lot queues and match disposals in chronological FIFO order."""

    def __init__(self) -> None:
        """Create an empty FIFO accounting ledger."""
        # Value: acquisition date, sequence number, lot
        self._lots_by_asset: dict[str, list[tuple[datetime, int, _FIFOLot]]] = {}
        # Track the next available sequence number
        self._next_sequence = 0

    def add_purchase(
        self,
        asset: str,
        acquisition_date: datetime,
        quantity: Decimal,
        unit_price: Decimal,
        fees: Decimal,
    ) -> None:
        """Record a purchase as a new open lot.

        Args:
            asset: Non-blank asset identifier.
            acquisition_date: Purchase timestamp.
            quantity: Strictly positive purchased quantity.
            unit_price: Non-negative acquisition price per unit.
            fees: Non-negative total acquisition fees.

        Raises:
            TypeError: If a value does not use the strict public API types.
            ValueError: If a financial or asset invariant is invalid.
        """
        self._validate_asset(asset, "asset")
        self._validate_datetime(acquisition_date, "acquisition_date")
        self._validate_positive_decimal(quantity, "quantity")
        self._validate_non_negative_decimal(unit_price, "unit_price")
        self._validate_non_negative_decimal(fees, "fees")
        self._validate_temporal_compatibility(asset, acquisition_date)

        lot = _FIFOLot(
            metadata=_LotMetadata(
                asset=asset,
                acquisition_date=acquisition_date,
                original_quantity=quantity,
                unit_price=unit_price,
                acquisition_fees=fees,
            ),
            remaining_quantity=quantity,
            remaining_acquisition_fees=fees,
            remaining_cost_basis=(quantity * unit_price) + fees,
        )
        self._push_lot(lot)

    def process_sale(
        self,
        asset: str,
        disposal_date: datetime,
        quantity: Decimal,
        unit_price: Decimal,
        fees: Decimal,
    ) -> list[FIFOMatch]:
        """Match a disposal against open purchase lots in FIFO order.

        State is updated only after the complete sale has been validated.

        Args:
            asset: Non-blank asset identifier.
            disposal_date: Disposal timestamp.
            quantity: Strictly positive quantity to dispose.
            unit_price: Non-negative disposal price per unit.
            fees: Non-negative total disposal fees.

        Returns:
            One immutable match per consumed purchase lot.

        Raises:
            TypeError: If a value does not use the strict public API types.
            ValueError: If inputs are invalid, insufficient holdings exist, or a lot is dated after the disposal.
            PostconditionError: If a critical accounting invariant is violated.
        """
        self._validate_asset(asset, "asset")
        self._validate_datetime(disposal_date, "disposal_date")
        self._validate_positive_decimal(quantity, "quantity")
        self._validate_non_negative_decimal(unit_price, "unit_price")
        self._validate_non_negative_decimal(fees, "fees")
        self._validate_temporal_compatibility(asset, disposal_date)

        queue = self._lots_by_asset.get(asset)
        if not queue:
            raise ValueError(f"No open purchase lots available for asset: {asset}")

        ordered_lots = [entry[2] for entry in sorted(queue)]
        total_available = sum((lot.remaining_quantity for lot in ordered_lots), Decimal("0"))
        if total_available < quantity:
            raise ValueError(
                f"Insufficient shares for sale of {asset}. Requested: {quantity}, Available: {total_available}"
            )

        allocations, matches = self._plan_sale(
            ordered_lots,
            asset=asset,
            disposal_date=disposal_date,
            quantity=quantity,
            unit_price=unit_price,
            fees=fees,
        )
        self._validate_sale_postconditions(asset, quantity, matches, allocations)

        for allocation in allocations:
            allocation.lot.remaining_quantity -= allocation.quantity
            allocation.lot.remaining_acquisition_fees -= allocation.acquisition_fees
            allocation.lot.remaining_cost_basis -= allocation.cost_basis

        self._remove_consumed_lots(asset)
        return matches

    def get_holdings(self, asset: str) -> list[FIFOHolding]:
        """Return the chronologically ordered open holdings for an asset.

        Args:
            asset: Non-blank asset identifier.

        Returns:
            Immutable holdings with strictly positive remaining quantities.

        Raises:
            TypeError: If ``asset`` is not a string.
            ValueError: If ``asset`` is blank.
            PostconditionError: If the internal queue contains an invalid lot.
        """
        self._validate_asset(asset, "asset")
        queue = self._lots_by_asset.get(asset, [])
        holdings: list[FIFOHolding] = []
        for _, _, lot in sorted(queue):
            assert_postcondition(
                lot.remaining_quantity > Decimal("0"),
                "Open FIFO queue contains a non-positive remaining quantity.",
                extra={"asset": asset, "remaining_quantity": str(lot.remaining_quantity)},
            )
            holdings.append(
                FIFOHolding(
                    acquisition_date=lot.acquisition_date,
                    remaining_quantity=lot.remaining_quantity,
                    original_quantity=lot.original_quantity,
                    unit_price=lot.unit_price,
                    acquisition_fees=lot.acquisition_fees,
                    cost_basis_remaining=lot.remaining_cost_basis,
                )
            )
        return holdings

    def process_merger(
        self,
        old_asset: str,
        new_asset: str,
        merger_date: datetime,
        exchange_ratio: Decimal,
    ) -> FIFOMergerResult:
        """Transfer open lots to a replacement asset without realizing a gain.

        Each replacement lot keeps the original acquisition date and exact
        remaining cost basis. Fractional replacement shares remain in the
        ledger; any cash-in-lieu disposal must be supplied separately.

        Args:
            old_asset: Non-blank identifier of the absorbed asset.
            new_asset: Non-blank identifier of the replacement asset.
            merger_date: Effective merger timestamp.
            exchange_ratio: Strictly positive replacement units per old unit.

        Returns:
            Immutable aggregate details of the completed transformation.

        Raises:
            TypeError: If a value does not use the strict public API types.
            ValueError: If merger inputs are invalid or predate a source acquisition.
            PostconditionError: If quantity or cost basis cannot be conserved.
        """
        self._validate_asset(old_asset, "old_asset")
        self._validate_asset(new_asset, "new_asset")
        self._validate_datetime(merger_date, "merger_date")
        self._validate_positive_decimal(exchange_ratio, "exchange_ratio")
        if old_asset == new_asset:
            raise ValueError("A merger must use distinct old_asset and new_asset identifiers")

        source_queue = self._lots_by_asset.get(old_asset)
        if not source_queue:
            return self._empty_merger_result(old_asset, new_asset, merger_date, exchange_ratio)

        self._validate_temporal_compatibility(old_asset, merger_date)
        source_lots = [entry[2] for entry in sorted(source_queue)]
        for lot in source_lots:
            if lot.acquisition_date > merger_date:
                raise ValueError("Merger date cannot precede an open source lot acquisition date")

        replacement_lots = [self._transform_lot(lot, new_asset, exchange_ratio) for lot in source_lots]
        for lot in replacement_lots:
            self._validate_temporal_compatibility(new_asset, lot.acquisition_date)

        total_old_quantity = sum((lot.remaining_quantity for lot in source_lots), Decimal("0"))
        total_new_quantity = sum((lot.remaining_quantity for lot in replacement_lots), Decimal("0"))
        old_cost_basis = sum((lot.remaining_cost_basis for lot in source_lots), Decimal("0"))
        new_cost_basis = sum((lot.remaining_cost_basis for lot in replacement_lots), Decimal("0"))
        assert_postcondition(
            total_new_quantity == total_old_quantity * exchange_ratio,
            "Merger replacement quantity is not conserved by the exchange ratio.",
            extra={"old_asset": old_asset, "new_asset": new_asset},
        )
        assert_postcondition(
            new_cost_basis == old_cost_basis,
            "Merger replacement cost basis is not conserved.",
            extra={"old_asset": old_asset, "new_asset": new_asset},
        )

        for lot in replacement_lots:
            self._push_lot(lot)
        del self._lots_by_asset[old_asset]

        whole_quantity = total_new_quantity.to_integral_value(rounding=ROUND_DOWN)
        return FIFOMergerResult(
            old_asset=old_asset,
            new_asset=new_asset,
            merger_date=merger_date,
            exchange_ratio=exchange_ratio,
            total_old_quantity=total_old_quantity,
            total_new_quantity=total_new_quantity,
            whole_quantity=whole_quantity,
            fractional_quantity=total_new_quantity - whole_quantity,
            transformed_lots=len(replacement_lots),
        )

    def _plan_sale(
        self,
        ordered_lots: list[_FIFOLot],
        *,
        asset: str,
        disposal_date: datetime,
        quantity: Decimal,
        unit_price: Decimal,
        fees: Decimal,
    ) -> tuple[list[_SaleAllocation], list[FIFOMatch]]:
        """Build all sale allocations without changing internal lot balances."""
        remaining_to_sell = quantity
        remaining_sale_fees = fees
        allocations: list[_SaleAllocation] = []
        matches: list[FIFOMatch] = []
        for lot in ordered_lots:
            if lot.acquisition_date > disposal_date:
                raise ValueError("Cannot match a disposal against a future-dated purchase lot")
            if remaining_to_sell == Decimal("0"):
                break
            matched_quantity = min(lot.remaining_quantity, remaining_to_sell)
            buy_fees = (matched_quantity / lot.remaining_quantity) * lot.remaining_acquisition_fees
            cost_basis = (matched_quantity / lot.remaining_quantity) * lot.remaining_cost_basis
            sell_fees = (matched_quantity / remaining_to_sell) * remaining_sale_fees
            sale_proceeds = matched_quantity * unit_price
            net_proceeds = sale_proceeds - sell_fees
            matches.append(
                FIFOMatch(
                    asset=asset,
                    acquisition_date=lot.acquisition_date,
                    disposal_date=disposal_date,
                    quantity=matched_quantity,
                    buy_unit_price=lot.unit_price,
                    sell_unit_price=unit_price,
                    buy_fees_pro_rata=buy_fees,
                    sell_fees_pro_rata=sell_fees,
                    cost_basis=cost_basis,
                    sale_proceeds=sale_proceeds,
                    net_proceeds=net_proceeds,
                    realized_gain=net_proceeds - cost_basis,
                )
            )
            allocations.append(
                _SaleAllocation(
                    lot=lot,
                    quantity=matched_quantity,
                    acquisition_fees=buy_fees,
                    cost_basis=cost_basis,
                )
            )
            remaining_to_sell -= matched_quantity
            remaining_sale_fees -= sell_fees
        return allocations, matches

    def _validate_sale_postconditions(
        self,
        asset: str,
        requested_quantity: Decimal,
        matches: list[FIFOMatch],
        allocations: list[_SaleAllocation],
    ) -> None:
        """Validate critical sale invariants before applying any state changes."""
        matched_quantity = sum((match.quantity for match in matches), Decimal("0"))
        assert_postcondition(
            matched_quantity == requested_quantity,
            "FIFO matches do not cover the requested disposal quantity.",
            extra={"asset": asset, "requested_quantity": str(requested_quantity)},
        )
        for match, allocation in zip(matches, allocations, strict=True):
            assert_postcondition(match.quantity > Decimal("0"), "FIFO match quantity must be positive.")
            assert_postcondition(match.cost_basis >= Decimal("0"), "FIFO match cost basis cannot be negative.")
            assert_postcondition(match.sale_proceeds >= Decimal("0"), "FIFO sale proceeds cannot be negative.")
            assert_postcondition(
                allocation.lot.remaining_quantity - allocation.quantity >= Decimal("0"),
                "FIFO allocation would create a negative remaining quantity.",
            )
            assert_postcondition(
                allocation.lot.remaining_acquisition_fees - allocation.acquisition_fees >= Decimal("0"),
                "FIFO allocation would create negative remaining acquisition fees.",
            )
            assert_postcondition(
                allocation.lot.remaining_cost_basis - allocation.cost_basis >= Decimal("0"),
                "FIFO allocation would create a negative remaining cost basis.",
            )

    def _transform_lot(self, lot: _FIFOLot, new_asset: str, exchange_ratio: Decimal) -> _FIFOLot:
        """Create one replacement lot while preserving its remaining cost basis."""
        replacement_quantity = lot.remaining_quantity * exchange_ratio
        return _FIFOLot(
            metadata=_LotMetadata(
                asset=new_asset,
                acquisition_date=lot.acquisition_date,
                original_quantity=replacement_quantity,
                unit_price=(lot.remaining_cost_basis - lot.remaining_acquisition_fees) / replacement_quantity,
                acquisition_fees=lot.remaining_acquisition_fees,
            ),
            remaining_quantity=replacement_quantity,
            remaining_acquisition_fees=lot.remaining_acquisition_fees,
            remaining_cost_basis=lot.remaining_cost_basis,
        )

    def _empty_merger_result(
        self, old_asset: str, new_asset: str, merger_date: datetime, exchange_ratio: Decimal
    ) -> FIFOMergerResult:
        """Return the documented no-op result for an asset with no open lots."""
        zero = Decimal("0")
        return FIFOMergerResult(
            old_asset=old_asset,
            new_asset=new_asset,
            merger_date=merger_date,
            exchange_ratio=exchange_ratio,
            total_old_quantity=zero,
            total_new_quantity=zero,
            whole_quantity=zero,
            fractional_quantity=zero,
            transformed_lots=0,
        )

    def _push_lot(self, lot: _FIFOLot) -> None:
        """Insert a validated internal lot into its asset's chronological heap."""
        queue = self._lots_by_asset.setdefault(lot.asset, [])
        heapq.heappush(queue, (lot.acquisition_date, self._next_sequence, lot))
        self._next_sequence += 1

    def _remove_consumed_lots(self, asset: str) -> None:
        """Discard fully consumed lots after a successful sale commit."""
        queue = self._lots_by_asset[asset]
        remaining_queue = [entry for entry in queue if entry[2].remaining_quantity > Decimal("0")]
        if remaining_queue:
            heapq.heapify(remaining_queue)
            self._lots_by_asset[asset] = remaining_queue
        else:
            del self._lots_by_asset[asset]

    def _validate_temporal_compatibility(self, asset: str, timestamp: datetime) -> None:
        """Reject naive/aware datetime mixes within a single asset queue."""
        queue = self._lots_by_asset.get(asset, [])
        if not queue:
            return
        existing_timestamp = queue[0][0]
        if self._is_aware(existing_timestamp) != self._is_aware(timestamp):
            raise ValueError(f"Cannot mix naive and timezone-aware datetimes for asset: {asset}")

    @staticmethod
    def _is_aware(timestamp: datetime) -> bool:
        """Return whether a datetime has a usable UTC offset."""
        return timestamp.tzinfo is not None and timestamp.utcoffset() is not None

    @staticmethod
    def _validate_asset(value: str, field_name: str) -> None:
        """Validate a strict non-blank asset identifier."""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a str")
        if not value.strip():
            raise ValueError(f"{field_name} must not be blank")

    @staticmethod
    def _validate_datetime(value: datetime, field_name: str) -> None:
        """Validate a strict datetime value."""
        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime")

    @staticmethod
    def _validate_positive_decimal(value: Decimal, field_name: str) -> None:
        """Validate a finite Decimal strictly greater than zero."""
        FIFOAccounting._validate_decimal_type(value, field_name)
        if not value.is_finite():
            raise ValueError(f"{field_name} must be finite")
        if value <= Decimal("0"):
            raise ValueError(f"{field_name} must be positive")

    @staticmethod
    def _validate_non_negative_decimal(value: Decimal, field_name: str) -> None:
        """Validate a finite Decimal greater than or equal to zero."""
        FIFOAccounting._validate_decimal_type(value, field_name)
        if not value.is_finite():
            raise ValueError(f"{field_name} must be finite")
        if value < Decimal("0"):
            raise ValueError(f"{field_name} cannot be negative")

    @staticmethod
    def _validate_decimal_type(value: Decimal, field_name: str) -> None:
        """Validate that a public financial input is a Decimal instance."""
        if not isinstance(value, Decimal):
            raise TypeError(f"{field_name} must be a Decimal")
