"""Persistence SQLModel database table entities."""

from datetime import datetime, timezone
from decimal import Decimal

from sqlmodel import Field, SQLModel


class TaxDocumentMetadata(SQLModel, table=True):
    """Chunked tax regulation and research document text and vector search metadata."""

    __tablename__ = "tax_document_metadata"  # type: ignore

    id: int = Field(primary_key=True)
    document_name: str
    document_sha: str = Field(..., index=True)
    jurisdiction: str | None = Field(default=None)  # 'italy', 'ireland', or None for cross-jurisdiction research
    source_type: str  # 'regulation' or 'research'
    confidence_level: str  # 'high', 'medium', or 'low'
    page_number: int
    text_content: str
    chunk_index: int
    parent_chunk_id: int | None = Field(default=None)
    parent_text_content: str | None = Field(default=None)
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))


class FinancialRecord(SQLModel, table=True):
    """Approved financial transaction ledger row stored in database."""

    __tablename__ = "financial_records"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    provider: str | None = Field(default=None)
    source_file_name: str | None = None
    source_file_sha: str | None = Field(default=None)
    event_timestamp: datetime | None = Field(default=None)
    ingestion_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    asset_type: str | None = Field(default=None)
    symbol: str | None = None
    isin: str | None = None
    asset_name: str | None = None
    action: str | None = Field(default=None)
    quantity: Decimal | None = Field(default=None, nullable=True)
    unit_price: Decimal | None = Field(default=None, nullable=True)
    currency: str = Field(default="EUR")
    fees: Decimal | None = Field(default=None, nullable=True)
    total_amount: Decimal | None = Field(default=Decimal("0.0"), nullable=True)
    fx_rate: Decimal | None = Field(default=None, nullable=True)
    local_total_amount: Decimal | None = Field(default=None, nullable=True)
    tax_year: int | None = Field(default=None, nullable=True)
    account_country: str | None = Field(default=None, nullable=True)
    additional_metadata: str | None = None
    verification_status: str = Field(default="pending_verification")
    consensus_log: str | None = None
    openfigi_detected: str | None = None


class StagedFinancialRecord(SQLModel, table=True):
    """Staging table for extracted records awaiting voter consensus approval or user review."""

    __tablename__ = "staged_financial_records"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    provider: str | None = Field(default=None)
    source_file_name: str | None = None
    source_file_sha: str | None = Field(default=None)
    event_timestamp: datetime | None = Field(default=None)
    ingestion_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    asset_type: str | None = Field(default=None)
    symbol: str | None = None
    isin: str | None = Field(default=None)
    asset_name: str | None = None
    action: str | None = Field(default=None)
    quantity: Decimal | None = Field(default=None, nullable=True)
    unit_price: Decimal | None = Field(default=None, nullable=True)
    currency: str = Field(default="EUR")
    fees: Decimal | None = Field(default=None, nullable=True)
    total_amount: Decimal | None = Field(default=None, nullable=True)
    fx_rate: Decimal | None = Field(default=None, nullable=True)
    local_total_amount: Decimal | None = Field(default=None, nullable=True)
    tax_year: int | None = Field(default=None, nullable=True)
    account_country: str | None = Field(default=None, nullable=True)
    additional_metadata: str | None = None
    verification_status: str = Field(default="pending_approval")
    approved_financial_record_id: int | None = Field(default=None, nullable=True)
    consensus_log: str | None = None
    openfigi_detected: str | None = None

    def get_missing_fields(self) -> list[str]:
        """Return list of required field names missing or invalid for approved ledger promotion."""
        missing: list[str] = []
        if not self.provider:
            missing.append("provider")
        if not self.account_country:
            missing.append("account_country")
        if self.event_timestamp is None:
            missing.append("event_timestamp")
        if not self.asset_type:
            missing.append("asset_type")
        if not self.action:
            missing.append("action")
        if self.total_amount is None:
            missing.append("total_amount")
        if self.tax_year is None:
            missing.append("tax_year")

        action_str = self.action.lower().strip() if self.action else ""
        is_non_trade = (
            action_str in ("dividend", "tax_payment", "salary_payout")
            or "dividend" in action_str
            or "ritenuta" in action_str
            or "tax" in action_str
            or "salary" in action_str
        )
        if not is_non_trade:
            if self.quantity is None:
                missing.append("quantity")
            if self.unit_price is None:
                missing.append("unit_price")

        return missing

    def is_approvable(self) -> bool:
        """Check if record has all mandatory fields complete and non-null for approval."""
        return len(self.get_missing_fields()) == 0


class IngestedSourceDocument(SQLModel, table=True):
    """Tracks source PDF documents ingested by the transaction pipeline to prevent duplicate processing."""

    __tablename__ = "ingested_source_documents"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    file_sha: str = Field(..., index=True, sa_column_kwargs={"unique": True})
    file_name: str
    provider: str
    account_country: str
    status: str = Field(..., description="'SUCCESS' or 'FAILED'")
    transaction_count: int = Field(default=0)
    error_message: str | None = Field(default=None)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AssetMerger(SQLModel, table=True):
    """Tracks corporate actions and ETF mergers (absorption of old ISIN into new ISIN).

    Note:
        Reconstruction engines use single-pass ISIN remapping. Multi-hop chained mergers
        (A -> B -> C) require individual or direct old-to-new mapping records.
    """

    __tablename__ = "asset_mergers"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    old_isin: str = Field(..., index=True)
    new_isin: str = Field(..., index=True)
    old_symbol: str | None = None
    new_symbol: str | None = None
    effective_date: datetime
    exchange_ratio: Decimal = Field(default=Decimal("1.0"))
    old_nav: Decimal | None = Field(default=None, nullable=True)
    new_nav: Decimal | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaxIncomeRecord(SQLModel, table=True):
    """Stores official tax income records (e.g. Irish EDS, Italian CU) with JSON payload."""

    __tablename__ = "tax_income_records"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    tax_year: int = Field(..., index=True)
    jurisdiction: str = Field(..., index=True)
    income_type: str = Field(...)
    source_document_sha: str | None = Field(default=None, nullable=True)
    payload_json: str = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

