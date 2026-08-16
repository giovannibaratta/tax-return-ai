"""Schemas for voter LLM extractions, consensus deliberation, and mismatch logs."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, field_validator, model_validator
from pydantic import Field as PydanticField

from backend.domain_models import AssetType, TransactionAction


class TransactionExtractionItem(BaseModel):
    """Schema for individual tax transactions extracted by Voter LLMs."""

    event_date: datetime = PydanticField(
        ...,
        description="ISO 8601 formatted datetime string (YYYY-MM-DDTHH:MM:SS) representing when transaction occurred.",
    )
    asset_type: AssetType = PydanticField(
        ...,
        description="Asset type: 'stock', 'etf', 'cash', 'tax_payment', 'salary', 'pension'.",
    )
    symbol: str | None = PydanticField(
        default=None,
        description="Ticker symbol or asset identifier (e.g., AAPL, VUAA, or null if cash/taxes).",
    )
    isin: str | None = PydanticField(
        default=None,
        description="International Securities Identification Number (ISIN) if available (e.g., US0378331005).",
    )
    action: TransactionAction = PydanticField(
        ...,
        description="Transaction action: 'buy', 'sell', 'dividend', 'tax_payment', 'salary_payout'.",
    )
    quantity: Decimal | None = PydanticField(
        default=None,
        description="Quantity of units/shares. Optional if not applicable.",
    )
    unit_price: Decimal | None = PydanticField(
        default=None,
        description="Price per unit in transaction currency.",
    )
    currency: str = PydanticField(
        default="EUR",
        description="Transaction currency code (e.g., EUR, USD, GBP).",
    )
    fees: Decimal | None = PydanticField(
        default=None,
        description="Total execution fees in transaction currency.",
    )
    total_amount: Decimal = PydanticField(
        ...,
        description="Total transaction amount in transaction currency.",
    )
    fx_rate: Decimal | None = PydanticField(
        default=None,
        description="Foreign exchange rate to local currency (EUR) on event date.",
    )
    asset_name: str | None = PydanticField(
        default=None,
        description="Friendly description/name of security mapped from statement.",
    )
    provider: str | None = PydanticField(
        default=None,
        description="Name of broker, bank, or tax authority (e.g., Directa, Interactive Brokers, Revenue).",
    )

    @field_validator("quantity", "unit_price", "fees", "total_amount", "fx_rate", mode="before")
    @classmethod
    def clean_numeric_field(cls, val: Any) -> Decimal | None:
        """Sanitizes raw LLM string float representations (e.g. European decimals, commas). Raises ValueError on invalid numbers."""
        if val is None or val == "":
            return None
        if isinstance(val, (int, float, Decimal)):
            return abs(Decimal(str(val)))
        s = str(val).replace("$", "").replace("€", "").replace("£", "").replace(" ", "").strip()
        if not s:
            return None

        if "," in s and "." in s:
            s = s.replace(",", "")
        elif "," in s:
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) == 3:
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")

        try:
            return abs(Decimal(s))
        except Exception as e:
            raise ValueError(f"Invalid decimal string representation '{val}': {e}") from e


class MismatchItem(BaseModel):
    """Details a voter mismatch detected at a specific candidate_records index."""

    index: int = PydanticField(
        description="0-indexed position pointing directly to the corresponding item in candidate_records list."
    )
    voter1: TransactionExtractionItem | None = None
    voter2: TransactionExtractionItem | None = None
    voter3: TransactionExtractionItem | None = None
    similarity_score: float | None = PydanticField(
        default=None, description="Calculated similarity percentage score between voter items (0.0 to 1.0)."
    )

    @model_validator(mode="after")
    def validate_mismatch_voters(self) -> "MismatchItem":
        """Ensures at least two voter extraction items are provided for comparison in a mismatch entry."""
        present_voters = sum(1 for v in (self.voter1, self.voter2, self.voter3) if v is not None)
        if present_voters < 2:
            raise ValueError("MismatchItem must specify extraction candidates from at least two voters.")
        return self


class ConsensusLog(BaseModel):
    """Schema for consensus verification logs stored in database."""

    version: str = PydanticField("1.0", description="Schema version identifier for database logs.")
    message: str | None = None
    error: str | None = None
    mismatches: list[MismatchItem] = PydanticField(default_factory=list[MismatchItem])
    raw_voter_1_records: list[TransactionExtractionItem] = PydanticField(
        default_factory=list[TransactionExtractionItem]
    )
    raw_voter_2_records: list[TransactionExtractionItem] = PydanticField(
        default_factory=list[TransactionExtractionItem]
    )
    raw_voter_3_records: list[TransactionExtractionItem] = PydanticField(
        default_factory=list[TransactionExtractionItem]
    )

    @model_validator(mode="after")
    def validate_consensus_payload(self) -> "ConsensusLog":
        """Validates that a ConsensusLog contains either explicit status messages, error logs, or voter records."""
        has_info = bool(self.message and self.message.strip()) or bool(self.error and self.error.strip())
        has_records = bool(
            self.raw_voter_1_records or self.raw_voter_2_records or self.raw_voter_3_records or self.mismatches
        )
        if not (has_info or has_records):
            raise ValueError(
                "ConsensusLog must contain at least one message, error, mismatch, or voter extraction record."
            )
        return self
