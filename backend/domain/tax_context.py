"""Tax calculation context for jurisdiction-specific computations."""

from datetime import date

from pydantic import BaseModel


class TaxContext(BaseModel):
    """Jurisdiction-specific calculation context for a tax year.

    Attributes:
        tax_jurisdiction: ISO 3166-1 alpha-2 tax filing country code.
        tax_year: Fiscal year for calculation.
        residency_start: First residency date, when residency began mid-year.
        residency_end: Last residency date, when residency ended mid-year.
    """

    tax_jurisdiction: str
    tax_year: int
    residency_start: date | None = None
    residency_end: date | None = None
