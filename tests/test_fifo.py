"""Tests for the strict FIFO accounting engine."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import cast

import pytest

from backend.services.accounting.fifo import FIFOAccounting


def _dt(value: str) -> datetime:
    """Build a naive test datetime from an ISO-8601 date string."""
    return datetime.fromisoformat(value)


@pytest.fixture
def accounting() -> FIFOAccounting:
    """Provide an empty FIFO ledger."""
    return FIFOAccounting()


def test_single_purchase_and_sale_returns_attribute_models(accounting: FIFOAccounting) -> None:
    # Given: A fully funded purchase lot
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("150"), Decimal("0"))

    # When: The entire lot is sold
    matches = accounting.process_sale("AAPL", _dt("2025-01-02"), Decimal("10"), Decimal("160"), Decimal("0"))

    # Then: The exact gain is returned through immutable attributes and no holding remains
    assert matches[0].acquisition_date == _dt("2025-01-01")
    assert matches[0].disposal_date == _dt("2025-01-02")
    assert matches[0].cost_basis == Decimal("1500")
    assert matches[0].sale_proceeds == Decimal("1600")
    assert matches[0].realized_gain == Decimal("100")
    assert accounting.get_holdings("AAPL") == []


def test_partial_sales_preserve_exact_fee_and_cost_basis(accounting: FIFOAccounting) -> None:
    # Given: A lot whose fee allocation produces repeating decimal values
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("3"), Decimal("100"), Decimal("1"))

    # When: The lot is sold in three one-unit disposals
    matches = [
        accounting.process_sale("AAPL", _dt(f"2025-01-0{day}"), Decimal("1"), Decimal("120"), Decimal("1"))[0]
        for day in range(2, 5)
    ]

    # Then: No in-engine rounding loses acquisition or disposal fees
    assert sum((match.buy_fees_pro_rata for match in matches), Decimal("0")) == Decimal("1")
    assert sum((match.sell_fees_pro_rata for match in matches), Decimal("0")) == Decimal("3")
    assert sum((match.cost_basis for match in matches), Decimal("0")) == Decimal("301")
    assert accounting.get_holdings("AAPL") == []


def test_same_timestamp_lots_use_insertion_order(accounting: FIFOAccounting) -> None:
    # Given: Two lots added at the same timestamp
    acquisition_date = _dt("2025-01-01")
    accounting.add_purchase("AAPL", acquisition_date, Decimal("5"), Decimal("100"), Decimal("0"))
    accounting.add_purchase("AAPL", acquisition_date, Decimal("5"), Decimal("110"), Decimal("0"))

    # When: A disposal crosses the lot boundary
    matches = accounting.process_sale("AAPL", _dt("2025-01-02"), Decimal("6"), Decimal("150"), Decimal("0"))

    # Then: The first inserted lot is consumed first
    assert [match.buy_unit_price for match in matches] == [Decimal("100"), Decimal("110")]
    assert [match.quantity for match in matches] == [Decimal("5"), Decimal("1")]


def test_reverse_date_insertion_still_matches_oldest_lot(accounting: FIFOAccounting) -> None:
    # Given: Lots recorded out of chronological order
    accounting.add_purchase("AAPL", _dt("2025-01-02"), Decimal("10"), Decimal("200"), Decimal("0"))
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("0"))

    # When: A sale is processed after both acquisitions
    matches = accounting.process_sale("AAPL", _dt("2025-01-03"), Decimal("5"), Decimal("150"), Decimal("0"))

    # Then: FIFO uses the earlier acquisition rather than insertion order
    assert matches[0].acquisition_date == _dt("2025-01-01")
    assert matches[0].buy_unit_price == Decimal("100")


def test_multiple_partial_sales_across_lots_remove_consumed_lots(accounting: FIFOAccounting) -> None:
    # Given: Two chronological purchase lots
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("0"))
    accounting.add_purchase("AAPL", _dt("2025-01-02"), Decimal("10"), Decimal("200"), Decimal("0"))

    # When: Consecutive sales fully consume the first lot and partially consume the second
    accounting.process_sale("AAPL", _dt("2025-01-03"), Decimal("5"), Decimal("150"), Decimal("0"))
    accounting.process_sale("AAPL", _dt("2025-01-04"), Decimal("8"), Decimal("150"), Decimal("0"))

    # Then: No zero-quantity lot remains and the second lot has the correct balance
    holdings = accounting.get_holdings("AAPL")
    assert len(holdings) == 1
    assert holdings[0].remaining_quantity == Decimal("7")
    assert all(entry[2].remaining_quantity > Decimal("0") for entry in accounting._lots_by_asset["AAPL"])


def test_multi_lot_fee_and_cost_basis_conservation(accounting: FIFOAccounting) -> None:
    # Given: Two lots with acquisition fees
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("10"))
    accounting.add_purchase("AAPL", _dt("2025-01-02"), Decimal("10"), Decimal("200"), Decimal("10"))

    # When: A sale spans both lots and carries disposal fees
    matches = accounting.process_sale("AAPL", _dt("2025-01-03"), Decimal("15"), Decimal("150"), Decimal("15"))

    # Then: Allocated fees and sold-plus-open cost basis are conserved exactly
    remaining_basis = accounting.get_holdings("AAPL")[0].cost_basis_remaining
    assert sum((match.buy_fees_pro_rata for match in matches), Decimal("0")) == Decimal("15")
    assert sum((match.sell_fees_pro_rata for match in matches), Decimal("0")) == Decimal("15")
    assert sum((match.cost_basis for match in matches), Decimal("0")) + remaining_basis == Decimal("3020")


def test_losses_and_zero_gains_are_reported(accounting: FIFOAccounting) -> None:
    # Given: A purchase lot at a known unit price
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("150"), Decimal("0"))

    # When: One partial disposal is a loss and the final disposal is break-even
    loss_match = accounting.process_sale("AAPL", _dt("2025-01-02"), Decimal("5"), Decimal("140"), Decimal("0"))[0]
    break_even_match = accounting.process_sale(
        "AAPL", _dt("2025-01-03"), Decimal("5"), Decimal("150"), Decimal("0")
    )[0]

    # Then: Both gain outcomes are preserved without coercion or rounding
    assert loss_match.realized_gain == Decimal("-50")
    assert break_even_match.realized_gain == Decimal("0")


@pytest.mark.parametrize(
    ("operation", "quantity", "unit_price", "fees"),
    [
        ("purchase", Decimal("-1"), Decimal("100"), Decimal("0")),
        ("purchase", Decimal("1"), Decimal("-100"), Decimal("0")),
        ("purchase", Decimal("1"), Decimal("100"), Decimal("-1")),
        ("sale", Decimal("0"), Decimal("150"), Decimal("0")),
        ("sale", Decimal("1"), Decimal("-150"), Decimal("0")),
        ("sale", Decimal("1"), Decimal("150"), Decimal("-1")),
    ],
)
def test_financial_inputs_must_respect_sign_invariants(
    accounting: FIFOAccounting,
    operation: str,
    quantity: Decimal,
    unit_price: Decimal,
    fees: Decimal,
) -> None:
    # Given: An accounting ledger with one valid lot for sale validations
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("0"))

    # When: A purchase or sale violates a financial sign invariant
    with pytest.raises(ValueError):
        if operation == "purchase":
            accounting.add_purchase("AAPL", _dt("2025-01-02"), quantity, unit_price, fees)
        else:
            accounting.process_sale("AAPL", _dt("2025-01-02"), quantity, unit_price, fees)

    # Then: The valid original lot is unaffected
    assert accounting.get_holdings("AAPL")[0].remaining_quantity == Decimal("10")


def test_asset_isolation_rejects_sale_without_matching_lots(accounting: FIFOAccounting) -> None:
    # Given: Holdings in one asset only
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("0"))

    # When: A different asset is sold
    with pytest.raises(ValueError, match="No open purchase lots"):
        accounting.process_sale("MSFT", _dt("2025-01-02"), Decimal("5"), Decimal("150"), Decimal("0"))

    # Then: The original asset remains unchanged
    assert accounting.get_holdings("AAPL")[0].remaining_quantity == Decimal("10")


def test_sale_failure_is_atomic(accounting: FIFOAccounting) -> None:
    # Given: An open purchase lot
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("2"))

    # When: A sale exceeds the available quantity
    with pytest.raises(ValueError, match="Insufficient shares"):
        accounting.process_sale("AAPL", _dt("2025-01-02"), Decimal("11"), Decimal("150"), Decimal("0"))

    # Then: The original holding remains unchanged
    holding = accounting.get_holdings("AAPL")[0]
    assert holding.remaining_quantity == Decimal("10")
    assert holding.cost_basis_remaining == Decimal("1002")


@pytest.mark.parametrize(
    ("value", "field_name"),
    [
        ("10", "quantity"),
        (10, "quantity"),
        (10.0, "quantity"),
        (Decimal("NaN"), "quantity"),
        (Decimal("Infinity"), "quantity"),
    ],
)
def test_purchase_rejects_non_strict_or_non_finite_decimal_inputs(
    accounting: FIFOAccounting, value: object, field_name: str
) -> None:
    # Given: An invalid public quantity value
    # When: It is supplied to the strict purchase API
    with pytest.raises((TypeError, ValueError), match=field_name):
        accounting.add_purchase("AAPL", _dt("2025-01-01"), cast(Decimal, value), Decimal("100"), Decimal("0"))

    # Then: No holding is recorded
    assert accounting.get_holdings("AAPL") == []


def test_purchase_rejects_blank_asset_and_datetime_mismatch(accounting: FIFOAccounting) -> None:
    # Given: A ledger with a naive-datetime asset queue
    accounting.add_purchase("AAPL", _dt("2025-01-01"), Decimal("1"), Decimal("100"), Decimal("0"))

    # When: Invalid identifiers and temporal conventions are used
    with pytest.raises(ValueError, match="blank"):
        accounting.add_purchase(" ", _dt("2025-01-01"), Decimal("1"), Decimal("100"), Decimal("0"))
    with pytest.raises(ValueError, match="naive"):
        accounting.add_purchase(
            "AAPL",
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            Decimal("1"),
            Decimal("100"),
            Decimal("0"),
        )

    # Then: The original queue is still intact
    assert len(accounting.get_holdings("AAPL")) == 1


def test_sale_rejects_future_purchase_without_mutating_state(accounting: FIFOAccounting) -> None:
    # Given: A lot acquired after the proposed disposal
    accounting.add_purchase("AAPL", _dt("2025-01-03"), Decimal("1"), Decimal("100"), Decimal("0"))

    # When: The earlier disposal is attempted
    with pytest.raises(ValueError, match="future-dated"):
        accounting.process_sale("AAPL", _dt("2025-01-02"), Decimal("1"), Decimal("150"), Decimal("0"))

    # Then: The future-dated lot remains available
    assert accounting.get_holdings("AAPL")[0].remaining_quantity == Decimal("1")


def test_merger_preserves_each_open_lot_and_fractional_quantity(accounting: FIFOAccounting) -> None:
    # Given: Two open source lots, including a fractional quantity
    accounting.add_purchase("OLD", _dt("2024-01-15"), Decimal("10"), Decimal("15"), Decimal("1"))
    accounting.add_purchase("OLD", _dt("2024-06-20"), Decimal("5.25"), Decimal("20"), Decimal("0.5"))

    # When: The source asset is replaced one-for-one
    result = accounting.process_merger("OLD", "NEW", _dt("2025-02-21"), Decimal("1"))

    # Then: Acquisition history, exact basis, and fractional holdings are retained
    holdings = accounting.get_holdings("NEW")
    assert accounting.get_holdings("OLD") == []
    assert [holding.acquisition_date for holding in holdings] == [_dt("2024-01-15"), _dt("2024-06-20")]
    assert sum((holding.cost_basis_remaining for holding in holdings), Decimal("0")) == Decimal("256.5")
    assert result.total_old_quantity == Decimal("15.25")
    assert result.total_new_quantity == Decimal("15.25")
    assert result.whole_quantity == Decimal("15")
    assert result.fractional_quantity == Decimal("0.25")


def test_non_one_merger_ratio_conserves_cost_basis(accounting: FIFOAccounting) -> None:
    # Given: A partially consumed source lot
    accounting.add_purchase("OLD", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("10"))
    accounting.process_sale("OLD", _dt("2025-01-02"), Decimal("4"), Decimal("110"), Decimal("0"))
    old_basis = accounting.get_holdings("OLD")[0].cost_basis_remaining

    # When: The remaining holding is exchanged at one half replacement unit per old unit
    result = accounting.process_merger("OLD", "NEW", _dt("2025-02-01"), Decimal("0.5"))

    # Then: The replacement lot has the same basis and scaled quantity
    holding = accounting.get_holdings("NEW")[0]
    assert holding.remaining_quantity == Decimal("3")
    assert holding.cost_basis_remaining == old_basis
    assert result.total_new_quantity == Decimal("3")


@pytest.mark.parametrize("invalid_ratio", [Decimal("0"), Decimal("-0.5")])
def test_merger_rejects_non_positive_exchange_ratios(accounting: FIFOAccounting, invalid_ratio: Decimal) -> None:
    # Given: An open source lot
    accounting.add_purchase("OLD", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("0"))

    # When: A non-positive exchange ratio is supplied
    with pytest.raises(ValueError, match="exchange_ratio must be positive"):
        accounting.process_merger("OLD", "NEW", _dt("2025-02-01"), invalid_ratio)

    # Then: The original holding remains open
    assert accounting.get_holdings("OLD")[0].remaining_quantity == Decimal("10")
    assert accounting.get_holdings("NEW") == []


def test_consecutive_mergers_preserve_basis_and_clear_intermediate_asset(accounting: FIFOAccounting) -> None:
    # Given: One source lot
    accounting.add_purchase("A", _dt("2025-01-01"), Decimal("10"), Decimal("100"), Decimal("0"))

    # When: It undergoes two consecutive tax-neutral mergers
    accounting.process_merger("A", "B", _dt("2025-02-01"), Decimal("2"))
    result = accounting.process_merger("B", "C", _dt("2025-03-01"), Decimal("0.5"))

    # Then: Quantity and cost basis return to their source values and intermediate queues are cleared
    holding = accounting.get_holdings("C")[0]
    assert result.total_old_quantity == Decimal("20")
    assert result.total_new_quantity == Decimal("10")
    assert result.transformed_lots == 1
    assert accounting.get_holdings("A") == []
    assert accounting.get_holdings("B") == []
    assert holding.remaining_quantity == Decimal("10")
    assert holding.cost_basis_remaining == Decimal("1000")


def test_merger_rejects_self_merger_and_earlier_effective_date(accounting: FIFOAccounting) -> None:
    # Given: An open source lot
    accounting.add_purchase("OLD", _dt("2025-01-15"), Decimal("10"), Decimal("100"), Decimal("0"))

    # When: Invalid merger identities or dates are requested
    with pytest.raises(ValueError, match="distinct"):
        accounting.process_merger("OLD", "OLD", _dt("2025-02-01"), Decimal("1"))
    with pytest.raises(ValueError, match="precede"):
        accounting.process_merger("OLD", "NEW", _dt("2025-01-01"), Decimal("1"))

    # Then: No source lot is moved
    assert len(accounting.get_holdings("OLD")) == 1
    assert accounting.get_holdings("NEW") == []


def test_empty_merger_is_noop_without_creating_destination(accounting: FIFOAccounting) -> None:
    # Given: No lots for the old asset
    # When: A merger is recorded
    result = accounting.process_merger("OLD", "NEW", _dt("2025-02-01"), Decimal("1"))

    # Then: The result is zero-valued and no destination queue exists
    assert result.total_old_quantity == Decimal("0")
    assert result.transformed_lots == 0
    assert accounting.get_holdings("NEW") == []
    assert "NEW" not in accounting._lots_by_asset
