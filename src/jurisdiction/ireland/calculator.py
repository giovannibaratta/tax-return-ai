"""Irish Income Tax & Pension Relief Calculator.

Statutory Regulations & References:
- Taxes Consolidation Act 1997 (TCA 1997), Section 790A: Earnings cap for pension contributions (€115,000).
- TCA 1997, Section 787B & Section 774(7): Age-related percentage limits for tax-relievable pension contributions.
- TCA 1997, Section 15: Standard Rate Cut-off Point (SRCOP) and income tax bands (Standard 20%, Higher 40%).
- TCA 1997, Section 461 & Section 472: Personal Tax Credit and Employee (PAYE) Tax Credit.
"""

from decimal import Decimal

from pydantic import BaseModel, Field

# Statutory Age Thresholds (TCA 1997 Section 787B)
AGE_UNDER_30 = 30
AGE_BAND_30_TO_39 = 39
AGE_BAND_40_TO_49 = 49
AGE_BAND_50_TO_54 = 54
AGE_BAND_55_TO_59 = 59

# Statutory Earnings Cap (TCA 1997 Section 790A)
EARNINGS_CAP_EUROS = Decimal("115000.00")


class PensionReliefResult(BaseModel):
    """Result container for Irish pension relief contribution limits under TCA 1997 Section 787B / 790A."""

    earnings_limit_applied: Decimal = Field(
        description="Capped net relevant earnings under TCA 1997 Section 790A (max €115,000)."
    )
    relief_percentage: Decimal = Field(
        description="Age-based maximum relief percentage allowed under TCA 1997 Section 787B."
    )
    max_allowable_contribution: Decimal = Field(description="Maximum tax-relievable pension contribution amount.")


class IrishPAYEResult(BaseModel):
    """Result container for Irish PAYE income tax computation under TCA 1997 Section 15."""

    gross_salary: Decimal = Field(description="Gross annual salary before deductions.")
    total_pension_contribution: Decimal = Field(description="Total annual pension contribution made.")
    allowed_pension_deduction: Decimal = Field(
        description="Tax-deductible pension contribution allowed under age/earnings limits."
    )
    unused_pension_relief: Decimal = Field(
        description="Pension contribution exceeding maximum allowable tax relief limit."
    )
    taxable_income: Decimal = Field(description="Net taxable income subject to PAYE.")
    gross_paye: Decimal = Field(description="Gross PAYE tax due before applying tax credits.")
    tax_credits: Decimal = Field(description="Total tax credits applied (e.g. Personal + PAYE credits).")
    net_paye_due: Decimal = Field(
        description="Final net PAYE income tax liability due after tax credits (min 0; not employee take-home pay)."
    )


def calculate_pension_relief_limit(
    age: int,
    net_relevant_earnings: int | float | str | Decimal,
) -> PensionReliefResult:
    """Calculates the maximum tax-relievable pension contribution limit for a given age and earnings in Ireland.

    Statutory Basis:
    - TCA 1997 Section 790A: Earnings cap for pension contributions is €115,000.
    - TCA 1997 Section 787B / 774(7): Age-related percentage limits:
        * Under 30: 15%
        * 30 to 39: 20%
        * 40 to 49: 25%
        * 50 to 54: 30%
        * 55 to 59: 35%
        * 60 and over: 40%

    Args:
        age: Age of the contributor in years.
        net_relevant_earnings: Net relevant annual earnings amount.

    Returns:
        PensionReliefResult model with capped earnings, relief percentage, and max allowable contribution.
    """
    earnings = Decimal(str(net_relevant_earnings))
    if earnings <= Decimal("0"):
        return PensionReliefResult(
            earnings_limit_applied=Decimal("0.00"),
            relief_percentage=Decimal("0.00"),
            max_allowable_contribution=Decimal("0.00"),
        )

    # Cap earnings at €115,000 per TCA 1997 Section 790A
    capped_earnings = min(earnings, EARNINGS_CAP_EUROS)

    # Determine percentage based on age per TCA 1997 Section 787B
    if age < AGE_UNDER_30:
        percentage = Decimal("0.15")
    elif age <= AGE_BAND_30_TO_39:
        percentage = Decimal("0.20")
    elif age <= AGE_BAND_40_TO_49:
        percentage = Decimal("0.25")
    elif age <= AGE_BAND_50_TO_54:
        percentage = Decimal("0.30")
    elif age <= AGE_BAND_55_TO_59:
        percentage = Decimal("0.35")
    else:
        percentage = Decimal("0.40")

    max_contribution = capped_earnings * percentage

    return PensionReliefResult(
        earnings_limit_applied=capped_earnings.quantize(Decimal("0.01")),
        relief_percentage=percentage,
        max_allowable_contribution=max_contribution.quantize(Decimal("0.01")),
    )


def calculate_irish_paye(
    gross_salary: int | float | str | Decimal,
    pension_contribution: int | float | str | Decimal,
    age: int,
    tax_credits: int | float | str | Decimal,
    srcop: int | float | str | Decimal,
) -> IrishPAYEResult:
    """Computes Irish PAYE income tax under TCA 1997 Section 15.

    Applies age-related pension relief to reduce taxable income before calculating tax bands.

    Statutory Basis:
    - Standard Rate (20%) applied to taxable income up to Standard Rate Cut-off Point (SRCOP).
    - Higher Rate (40%) applied to taxable income exceeding SRCOP.
    - Tax Credits (TCA 1997 Section 461/472) directly reduce gross PAYE liability.

    Args:
        gross_salary: Gross annual earnings.
        pension_contribution: Annual pension contribution amount.
        age: Age of taxpayer (used to determine pension relief cap).
        tax_credits: Total annual tax credits applicable.
        srcop: Standard Rate Cut-off Point (SRCOP) threshold.

    Returns:
        IrishPAYEResult model with full breakdown of taxable income and net tax liability.
    """
    gross = Decimal(str(gross_salary))
    pension = Decimal(str(pension_contribution))
    credits_val = Decimal(str(tax_credits))
    srcop_val = Decimal(str(srcop))

    # 1. Pension Relief Check
    relief_info = calculate_pension_relief_limit(age, gross)
    max_relievable = relief_info.max_allowable_contribution

    allowed_deduction = min(pension, max_relievable)
    unused_pension_relief = max(Decimal("0.00"), pension - max_relievable)
    taxable_income = max(Decimal("0.00"), gross - allowed_deduction)

    # 2. PAYE Tax Band Calculations
    standard_taxable = min(taxable_income, srcop_val)
    higher_taxable = max(Decimal("0.00"), taxable_income - srcop_val)

    gross_tax = (standard_taxable * Decimal("0.20")) + (higher_taxable * Decimal("0.40"))
    net_paye = max(Decimal("0.00"), gross_tax - credits_val)

    return IrishPAYEResult(
        gross_salary=gross.quantize(Decimal("0.01")),
        total_pension_contribution=pension.quantize(Decimal("0.01")),
        allowed_pension_deduction=allowed_deduction.quantize(Decimal("0.01")),
        unused_pension_relief=unused_pension_relief.quantize(Decimal("0.01")),
        taxable_income=taxable_income.quantize(Decimal("0.01")),
        gross_paye=gross_tax.quantize(Decimal("0.01")),
        tax_credits=credits_val.quantize(Decimal("0.01")),
        net_paye_due=net_paye.quantize(Decimal("0.01")),
    )
