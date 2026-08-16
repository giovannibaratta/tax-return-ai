"""Base tax computation result models."""

from decimal import Decimal

from pydantic import BaseModel


class TaxResult(BaseModel):
    """Jurisdiction-agnostic tax computation result envelope.

    Attributes:
        tax_jurisdiction: ISO 3166-1 alpha-2 tax filing country code.
        tax_year: Fiscal year covered by result.
        total_taxable_gain: Net taxable gain after applicable rules.
        total_tax_due: Computed tax liability for period.
    """

    tax_jurisdiction: str
    tax_year: int
    total_taxable_gain: Decimal
    total_tax_due: Decimal
