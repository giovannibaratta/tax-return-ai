"""Tax engine protocol for multi-jurisdiction dispatch."""

from typing import Protocol

from backend.domain.tax_context import TaxContext
from backend.domain.tax_result import TaxResult
from backend.domain_models import TradeRecord


class TaxEngine(Protocol):
    """Define interface for jurisdiction-specific tax engines."""

    def calculate(self, transactions: list[TradeRecord], context: TaxContext) -> TaxResult:
        """Compute tax liability for pre-loaded trade data.

        Args:
            transactions: Canonical trades to evaluate.
            context: Jurisdiction-specific tax context.

        Returns:
            Jurisdiction-specific tax result.
        """
        ...
