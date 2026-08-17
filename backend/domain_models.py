"""Strict domain models, extraction schemas, and discriminated unions for tax processing."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, TypeAdapter, field_validator, model_validator
from pydantic import Field as PydanticField

from backend.db_models import FinancialRecord, TaxIncomeRecord


class PostconditionError(ValueError):
    """Exception raised when a defensive post-condition validation fails in critical computations."""

    def __init__(self, message: str, extra: dict[str, object] | None = None) -> None:
        self.message = message
        self.extra = extra or {}
        full_msg = message
        if self.extra:
            formatted_extra = ", ".join(f"{k}={v}" for k, v in self.extra.items())
            full_msg = f"{message} | Context: [{formatted_extra}]"
        super().__init__(full_msg)


def assert_postcondition(
    condition: bool,
    message: str,
    extra: dict[str, object] | None = None,
) -> None:
    """Assert a defensive post-condition check for critical calculations.

    Args:
        condition: Boolean expression that MUST evaluate to True.
        message: Concise failure explanation message.
        extra: Optional context dictionary detailing intermediate calculation parameters.

    Raises:
        PostconditionError: If condition evaluates to False.
    """
    if not condition:
        raise PostconditionError(message=message, extra=extra)


class SourceType(str, Enum):
    """Source provenance category for ingested tax documents."""

    REGULATION = "regulation"
    RESEARCH = "research"


class ConfidenceLevel(str, Enum):
    """Confidence rating reflecting source authority and verification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RegulationChunkInput(BaseModel):
    """Domain input for regulation chunks. Jurisdiction is mandatory."""

    document_name: str
    document_sha: str
    jurisdiction: str  # required — regulations always belong to a jurisdiction
    page_number: int
    text_content: str
    chunk_index: int
    embedding: list[float]
    parent_text_content: str | None = None
    source_type: Literal[SourceType.REGULATION] = SourceType.REGULATION
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH


class ResearchChunkInput(BaseModel):
    """Domain input for research chunks. Jurisdiction is optional."""

    document_name: str
    document_sha: str
    jurisdiction: str | None = None  # optional — research may span jurisdictions
    page_number: int
    text_content: str
    chunk_index: int
    embedding: list[float]
    parent_text_content: str | None = None
    source_type: Literal[SourceType.RESEARCH] = SourceType.RESEARCH
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM


class IngestionDocumentSummary(BaseModel):
    """Summary of an ingested regulation or research document, returned by DatabaseManager queries."""

    document_name: str
    document_sha: str
    jurisdiction: str | None = None
    chunk_count: int
    source_type: str


class DocumentPageInfo(BaseModel):
    """Concatenated document page details."""

    document_name: str
    jurisdiction: str | None = None
    page_number: int
    total_pages: int
    text_content: str
    chunk_count: int
    has_previous_page: bool
    has_next_page: bool


class DocumentMetadata(BaseModel):
    """Aggregated metadata for an entire document."""

    document_name: str
    jurisdiction: str | None = None
    total_chunks: int
    page_range: str
    document_sha: str
    source_type: str


class DocumentStatistics(BaseModel):
    """Size metrics and statistics for an entire document."""

    document_name: str
    jurisdiction: str | None = None
    total_chunks: int
    start_page: int
    end_page: int
    total_pages: int


class IngestionStatus(str, Enum):
    """Status of source document ingestion in the transaction pipeline."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AssetType(str, Enum):
    """Supported asset classes across financial records."""

    STOCK = "stock"
    ETF = "etf"
    CASH = "cash"
    TAX_PAYMENT = "tax_payment"
    SALARY = "salary"
    PENSION = "pension"


class TransactionAction(str, Enum):
    """Supported transaction actions across financial records."""

    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    TAX_PAYMENT = "tax_payment"
    SALARY_PAYOUT = "salary_payout"
    CORPORATE_ACTION = "corporate_action"
    """Non-taxable corporate structural events (e.g. stock splits like NVDA 10:1, ETF ISIN mergers, spinoffs)."""


class VerificationStatus(str, Enum):
    """Status of financial record verification in the ingestion pipeline."""

    PENDING_VERIFICATION = "pending_verification"
    PENDING_APPROVAL = "pending_approval"
    VERIFIED = "verified"
    """Automatically verified by multi-voter LLM consensus without requiring user review."""
    APPROVED = "approved"
    """Manually reviewed and approved by the user in the UI staging workspace."""
    ESCALATED_TO_USER = "escalated_to_user"


class AssetIdentity(BaseModel):
    """Encapsulates core asset identification attributes (symbol, ISIN, asset name)."""

    symbol: str | None = None
    isin: str | None = None
    asset_name: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "AssetIdentity":
        """Ensures at least one of symbol, isin, or asset_name is provided."""
        has_symbol = bool(self.symbol and self.symbol.strip())
        has_isin = bool(self.isin and self.isin.strip())
        has_name = bool(self.asset_name and self.asset_name.strip())
        if not (has_symbol or has_isin or has_name):
            raise ValueError("AssetIdentity must provide at least one identifier: 'symbol', 'isin', or 'asset_name'.")
        return self

    @property
    def canonical_id(self) -> str:
        """Returns canonical asset identifier string in priority order: ISIN > symbol > asset_name."""
        if self.isin and self.isin.strip():
            return self.isin.strip()
        if self.symbol and self.symbol.strip():
            return self.symbol.strip()
        if self.asset_name and self.asset_name.strip():
            return self.asset_name.strip()
        raise ValueError("AssetIdentity missing all identifiers.")


class BaseStrictRecord(BaseModel):
    """Base class for strictly-typed financial transaction headers."""

    id: int
    provider: str
    source_file_sha: str | None = None
    event_timestamp: datetime
    ingestion_timestamp: datetime
    currency: str
    total_amount: Decimal
    fx_rate: Decimal
    local_total_amount: Decimal
    tax_year: int
    account_country: str
    verification_status: VerificationStatus

    def _base_to_raw_record(self, action: str) -> FinancialRecord:
        """Construct raw FinancialRecord entity with base header attributes."""
        verif_str = self.verification_status.value

        return FinancialRecord(
            id=self.id,
            provider=self.provider,
            source_file_sha=self.source_file_sha,
            event_timestamp=self.event_timestamp,
            ingestion_timestamp=self.ingestion_timestamp,
            currency=self.currency,
            total_amount=self.total_amount,
            fx_rate=self.fx_rate,
            local_total_amount=self.local_total_amount,
            tax_year=self.tax_year,
            account_country=self.account_country,
            verification_status=verif_str,
            action=action,
        )

    def to_raw(self) -> FinancialRecord:
        """Convert strict domain record back to SQLModel database entity for persistence.

        Raises:
            NotImplementedError: BaseStrictRecord is abstract; concrete record subclasses must implement to_raw.
        """
        raise NotImplementedError("to_raw() must be implemented by concrete record subclasses.")

    @classmethod
    def from_raw(cls, record: FinancialRecord) -> "StrictFinancialRecord":
        """Convert a raw FinancialRecord to an action-discriminated strict domain record.

        Raises:
            ValueError: If mandatory fields are missing, invalid, or cannot be mapped strictly.
        """
        if record.id is None:
            raise ValueError("FinancialRecord missing mandatory 'id'.")
        if not record.provider:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'provider'.")
        if not record.account_country:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'account_country'.")
        if record.event_timestamp is None:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'event_timestamp'.")
        if not record.currency:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'currency'.")
        if record.total_amount is None:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'total_amount'.")
        if record.fx_rate is None:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'fx_rate'.")
        if record.local_total_amount is None:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'local_total_amount'.")
        if record.tax_year is None:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'tax_year'.")
        if not record.verification_status:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'verification_status'.")
        if not record.action:
            raise ValueError(f"FinancialRecord {record.id} missing mandatory 'action'.")

        try:
            verif_status_enum = VerificationStatus(record.verification_status.lower().strip())
        except ValueError:
            raise ValueError(
                f"FinancialRecord {record.id} has invalid verification_status '{record.verification_status}'."
            )

        try:
            action_enum = TransactionAction(record.action.lower().strip())
        except ValueError:
            raise ValueError(f"FinancialRecord {record.id} has invalid or unmapped action '{record.action}'.")

        if action_enum in (TransactionAction.BUY, TransactionAction.SELL):
            if record.quantity is None:
                raise ValueError(f"Trade record {record.id} missing mandatory 'quantity'.")
            if record.quantity <= Decimal("0"):
                raise ValueError(f"Trade record {record.id} has non-positive quantity {record.quantity}.")
            if record.unit_price is None:
                raise ValueError(f"Trade record {record.id} missing mandatory 'unit_price'.")
            if record.fees is None:
                raise ValueError(f"Trade record {record.id} missing mandatory 'fees'.")
            if not record.asset_type:
                raise ValueError(f"Trade record {record.id} missing mandatory 'asset_type'.")
            try:
                asset_type_enum = AssetType(record.asset_type.lower().strip())
            except ValueError:
                raise ValueError(f"Trade record {record.id} has invalid asset_type '{record.asset_type}'.")

            identity = AssetIdentity(symbol=record.symbol, isin=record.isin, asset_name=record.asset_name)

            return TradeRecord(
                id=record.id,
                provider=record.provider,
                source_file_sha=record.source_file_sha,
                event_timestamp=record.event_timestamp,
                ingestion_timestamp=record.ingestion_timestamp,
                currency=record.currency,
                total_amount=record.total_amount,
                fx_rate=record.fx_rate,
                local_total_amount=record.local_total_amount,
                tax_year=record.tax_year,
                account_country=record.account_country,
                verification_status=verif_status_enum,
                action=action_enum,
                asset_type=asset_type_enum,
                identity=identity,
                quantity=record.quantity,
                unit_price=record.unit_price,
                fees=record.fees,
            )
        elif action_enum == TransactionAction.DIVIDEND:
            if not record.asset_type:
                raise ValueError(f"Dividend record {record.id} missing mandatory 'asset_type'.")
            try:
                asset_type_enum = AssetType(record.asset_type.lower().strip())
            except ValueError:
                raise ValueError(f"Dividend record {record.id} has invalid asset_type '{record.asset_type}'.")

            identity = AssetIdentity(symbol=record.symbol, isin=record.isin, asset_name=record.asset_name)
            return DividendRecord(
                id=record.id,
                provider=record.provider,
                source_file_sha=record.source_file_sha,
                event_timestamp=record.event_timestamp,
                ingestion_timestamp=record.ingestion_timestamp,
                currency=record.currency,
                total_amount=record.total_amount,
                fx_rate=record.fx_rate,
                local_total_amount=record.local_total_amount,
                tax_year=record.tax_year,
                account_country=record.account_country,
                verification_status=verif_status_enum,
                action=action_enum,
                asset_type=asset_type_enum,
                identity=identity,
            )
        elif action_enum == TransactionAction.TAX_PAYMENT:
            return TaxPaymentRecord(
                id=record.id,
                provider=record.provider,
                source_file_sha=record.source_file_sha,
                event_timestamp=record.event_timestamp,
                ingestion_timestamp=record.ingestion_timestamp,
                currency=record.currency,
                total_amount=record.total_amount,
                fx_rate=record.fx_rate,
                local_total_amount=record.local_total_amount,
                tax_year=record.tax_year,
                account_country=record.account_country,
                verification_status=verif_status_enum,
                action=action_enum,
            )
        else:
            raise ValueError(f"FinancialRecord {record.id} has unhandled action '{action_enum}'.")


class SecurityRecord(BaseStrictRecord):
    """Base class for financial records associated with a traded security or fund."""

    asset_type: AssetType
    identity: AssetIdentity

    @model_validator(mode="before")
    @classmethod
    def _construct_identity(cls, data: Any) -> Any:
        if isinstance(data, dict) and "identity" not in data:
            d: dict[str, Any] = data  # pyright: ignore [reportUnknownVariableType]
            raw_sym = d.get("symbol")
            raw_isin = d.get("isin")
            raw_name = d.get("asset_name")
            sym_str = str(raw_sym) if isinstance(raw_sym, str) else None
            isin_str = str(raw_isin) if isinstance(raw_isin, str) else None
            name_str = str(raw_name) if isinstance(raw_name, str) else None
            d["identity"] = AssetIdentity(symbol=sym_str, isin=isin_str, asset_name=name_str)
        return data  # pyright: ignore [reportUnknownVariableType]

    @property
    def symbol(self) -> str | None:
        return self.identity.symbol

    @property
    def isin(self) -> str | None:
        return self.identity.isin

    @property
    def asset_name(self) -> str | None:
        return self.identity.asset_name

    @property
    def asset_identifier(self) -> str:
        return self.identity.canonical_id

    def _security_to_raw_record(self, action: str) -> FinancialRecord:
        """Construct raw FinancialRecord entity with security attributes."""
        record = self._base_to_raw_record(action)
        record.asset_type = self.asset_type.value
        record.symbol = self.identity.symbol
        record.isin = self.identity.isin
        record.asset_name = self.identity.asset_name
        return record


class TradeRecord(SecurityRecord):
    """Strict model for stock/ETF purchase or sale transactions."""

    action: Literal[TransactionAction.BUY, TransactionAction.SELL]
    quantity: Decimal = PydanticField(gt=Decimal("0"), description="Trade quantity must be strictly positive (> 0).")
    unit_price: Decimal = PydanticField(ge=Decimal("0"), description="Unit price must be non-negative (>= 0).")
    fees: Decimal = PydanticField(ge=Decimal("0"), description="Transaction fees must be non-negative (>= 0).")

    @field_validator("quantity")
    @classmethod
    def validate_positive_quantity(cls, v: Decimal) -> Decimal:
        if v <= Decimal("0"):
            raise ValueError(f"TradeRecord quantity must be strictly positive (> 0), got {v}.")
        return v

    def to_raw(self) -> FinancialRecord:
        """Convert strict trade record back to SQLModel database entity for persistence.

        Returns:
            FinancialRecord SQLModel table entity.
        """
        act_str = self.action.value
        raw_record = self._security_to_raw_record(act_str)
        raw_record.quantity = self.quantity
        raw_record.unit_price = self.unit_price
        raw_record.fees = self.fees
        return raw_record


class DividendRecord(SecurityRecord):
    """Strict model for dividend payouts on securities."""

    action: Literal[TransactionAction.DIVIDEND]

    def to_raw(self) -> FinancialRecord:
        """Convert strict dividend record back to SQLModel database entity for persistence.

        Returns:
            FinancialRecord SQLModel table entity.
        """
        act_str = self.action.value
        return self._security_to_raw_record(act_str)


class TaxPaymentRecord(BaseStrictRecord):
    """Strict model for non-security tax payments or withholdings (e.g. F24)."""

    action: Literal[TransactionAction.TAX_PAYMENT]

    def to_raw(self) -> FinancialRecord:
        """Convert strict tax payment record back to SQLModel database entity for persistence.

        Returns:
            FinancialRecord SQLModel table entity.
        """
        act_str = self.action.value
        return self._base_to_raw_record(act_str)


StrictFinancialRecord = Annotated[
    TradeRecord | DividendRecord | TaxPaymentRecord,
    PydanticField(discriminator="action"),
]


class IrishEmploymentDetailSummaryPayload(BaseModel):
    """Payload representing an Irish Revenue Employment Detail Summary (EDS / P60)."""

    income_type: Literal["irish_employment_detail_summary"] = "irish_employment_detail_summary"
    tax_year: int
    employer_name: str
    employer_registration_number: str | None = None
    employment_id: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    gross_pay_eur: Decimal
    income_tax_paid_eur: Decimal
    usc_paid_eur: Decimal
    prsi_paid_eur: Decimal
    employer_prsi_paid_eur: Decimal | None = None
    prsi_class: str | None = None
    prsi_weeks: int | None = None
    lpt_deducted_eur: Decimal | None = None


IncomePayload = Annotated[
    IrishEmploymentDetailSummaryPayload,
    PydanticField(discriminator="income_type"),
]


class StrictTaxIncomeRecord(BaseModel):
    """Domain model for a verified tax income record with strongly typed payload."""

    id: int | None = None
    tax_year: int
    jurisdiction: str
    income_type: str
    source_document_sha: str | None = None
    payload: IncomePayload
    created_at: datetime

    @classmethod
    def from_raw(cls, raw: TaxIncomeRecord) -> "StrictTaxIncomeRecord":
        """Convert from raw database SQLModel entity."""
        adapter: TypeAdapter[IncomePayload] = TypeAdapter(IncomePayload)
        payload_obj: IncomePayload = adapter.validate_json(raw.payload_json)
        return cls(
            id=raw.id,
            tax_year=raw.tax_year,
            jurisdiction=raw.jurisdiction,
            income_type=raw.income_type,
            source_document_sha=raw.source_document_sha,
            payload=payload_obj,
            created_at=raw.created_at,
        )

    def to_raw(self) -> TaxIncomeRecord:
        """Convert domain model to raw SQLModel entity."""
        return TaxIncomeRecord(
            id=self.id,
            tax_year=self.tax_year,
            jurisdiction=self.jurisdiction,
            income_type=self.income_type,
            source_document_sha=self.source_document_sha,
            payload_json=self.payload.model_dump_json(),
            created_at=self.created_at,
        )

