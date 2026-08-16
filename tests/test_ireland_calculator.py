from decimal import Decimal

import pytest

from src.jurisdiction.ireland.calculator import (
    calculate_irish_paye,
    calculate_pension_relief_limit,
)


@pytest.mark.parametrize(
    "age, income, expected_limit, expected_percentage, expected_applied",
    [
        # Given: Various age bands and incomes
        (25, "50000", "7500.00", "0.15", "50000.00"),
        (29, "100000", "15000.00", "0.15", "100000.00"),  # Age 29 (under 30 boundary) -> 15%
        (30, "100000", "20000.00", "0.20", "100000.00"),  # Age 30 (band 30-39 lower boundary) -> 20%
        (35, "120000", "23000.00", "0.20", "115000.00"),  # Age 35, capped at 115000
        (35, "115000", "23000.00", "0.20", "115000.00"),  # Exact €115,000 cap boundary
        (35, "115001", "23000.00", "0.20", "115000.00"),  # Exceeding €115,000 cap by €1
        (39, "100000", "20000.00", "0.20", "100000.00"),  # Age 39 (band 30-39 upper boundary) -> 20%
        (40, "80000", "20000.00", "0.25", "80000.00"),  # Age 40 (band 40-49) -> 25%
        (49, "80000", "20000.00", "0.25", "80000.00"),  # Age 49 (upper boundary 40-49) -> 25%
        (52, "80000", "24000.00", "0.30", "80000.00"),  # Age 52 (band 50-54) -> 30%
        (54, "80000", "24000.00", "0.30", "80000.00"),  # Age 54 (upper boundary 50-54) -> 30%
        (55, "80000", "28000.00", "0.35", "80000.00"),  # Age 55 (band 55-59) -> 35%
        (59, "80000", "28000.00", "0.35", "80000.00"),  # Age 59 (upper boundary 55-59) -> 35%
        (60, "80000", "32000.00", "0.40", "80000.00"),  # Age 60 (band 60+) -> 40%
        (70, "80000", "32000.00", "0.40", "80000.00"),  # Age 70 (above 60) -> 40%
        (35, "0", "0.00", "0.00", "0.00"),  # Zero earnings
        (35, "-1000", "0.00", "0.00", "0.00"),  # Negative earnings
    ],
)
def test_pension_relief_limit(
    age: int,
    income: str,
    expected_limit: str,
    expected_percentage: str,
    expected_applied: str,
) -> None:
    # When: Calculating pension relief limit
    res = calculate_pension_relief_limit(age, Decimal(income))

    # Then: The limits, percentages and earnings applied are correct
    assert res.max_allowable_contribution == Decimal(expected_limit)
    assert res.relief_percentage == Decimal(expected_percentage)
    assert res.earnings_limit_applied == Decimal(expected_applied)


def test_irish_paye_calculation_no_pension() -> None:
    # Given: Gross salary €60,000, no pension, age 35, srcop €44,000, credits €4,000
    # When: Calculating PAYE
    res = calculate_irish_paye(
        Decimal("60000"),
        pension_contribution=0,
        age=35,
        tax_credits=Decimal("4000.00"),
        srcop=Decimal("44000.00"),
    )

    # Then: Gross paye and net paye are correct
    assert res.gross_paye == Decimal("15200.00")
    assert res.net_paye_due == Decimal("11200.00")


def test_irish_paye_calculation_with_pension() -> None:
    # Given: Gross salary €60,000, pension contribution €15,000, age 35, srcop €44,000, credits €4,000
    # When: Calculating PAYE
    res = calculate_irish_paye(
        Decimal("60000"),
        pension_contribution=15000,
        age=35,
        tax_credits=Decimal("4000.00"),
        srcop=Decimal("44000.00"),
    )

    # Then: Allowed pension, taxable income, and net paye are computed with pension taken into account
    assert res.allowed_pension_deduction == Decimal("12000.00")
    assert res.unused_pension_relief == Decimal("3000.00")
    assert res.taxable_income == Decimal("48000.00")
    assert res.net_paye_due == Decimal("6400.00")


def test_irish_paye_srcop_exact_boundary() -> None:
    # Given: Taxable income exactly at SRCOP €44,000
    # When: Calculating PAYE
    res = calculate_irish_paye(
        Decimal("44000.00"),
        pension_contribution=0,
        age=30,
        tax_credits=Decimal("4000.00"),
        srcop=Decimal("44000.00"),
    )

    # Then: All income taxed at standard 20%, 0 at higher rate
    assert res.taxable_income == Decimal("44000.00")
    assert res.gross_paye == Decimal("8800.00")
    assert res.net_paye_due == Decimal("4800.00")


def test_irish_paye_credits_exceed_gross_tax() -> None:
    # Given: Low income where credits exceed gross tax liability
    # When: Calculating PAYE
    res = calculate_irish_paye(
        Decimal("15000.00"),
        pension_contribution=0,
        age=25,
        tax_credits=Decimal("4000.00"),
        srcop=Decimal("44000.00"),
    )

    # Then: Gross tax is €3,000, net PAYE is floored at €0 (non-refundable tax credit)
    assert res.gross_paye == Decimal("3000.00")
    assert res.net_paye_due == Decimal("0.00")
