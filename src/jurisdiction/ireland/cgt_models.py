"""Irish Capital Gains Tax models for computation engine.

Pydantic and SQLModel definitions for tax regime classification,
taxpayer profile, remittance tracking, disposal inputs/outputs,
and FIFO lot match results.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlmodel import Field, SQLModel


class IrishTaxRegime(str, Enum):
    """Irish tax regime classification for financial assets.

    Determines applicable tax rate, loss offset rules, annual exemption
    eligibility, and deemed disposal applicability per TCA 1997.
    """

    CGT_STANDARD = "cgt_standard"
    EXIT_TAX = "exit_tax"
    OFFSHORE_DISTRIBUTING = "offshore_distributing"
    OFFSHORE_NON_DISTRIBUTING = "offshore_non_distributing"
    ETC_COMMODITY = "etc_commodity"


class ResidencyType(str, Enum):
    """Irish tax residency classification for a given tax year.

    Determines worldwide vs remittance-based taxation scope.
    """

    RESIDENT_DOMICILED = "resident_domiciled"
    RESIDENT_NON_DOMICILED = "resident_non_domiciled"
    NON_RESIDENT = "non_resident"


def infer_residency_type(fiscal_residence_country: str, domicile_country: str) -> ResidencyType:
    """Infer Irish residency classification from fiscal residence and domicile countries.

    Rules:
    - Fiscal residence != 'IE' -> NON_RESIDENT
    - Fiscal residence == 'IE' & Domicile == 'IE' -> RESIDENT_DOMICILED
    - Fiscal residence == 'IE' & Domicile != 'IE' -> RESIDENT_NON_DOMICILED
    """
    if fiscal_residence_country.strip().upper() != "IE":
        return ResidencyType.NON_RESIDENT
    if domicile_country.strip().upper() == "IE":
        return ResidencyType.RESIDENT_DOMICILED
    return ResidencyType.RESIDENT_NON_DOMICILED


def infer_tax_regime(
    *,
    is_ucits: bool,
    is_etc: bool,
    is_offshore_distributing: bool,
    is_direct_equity_or_crypto: bool,
) -> IrishTaxRegime:
    """Infer Irish tax regime from asset characteristics without silent fallbacks.

    Statutory Basis (TCA 1997):
    - UCITS ETF (EU/Irish domiciled) -> EXIT_TAX (41%/38% exit tax, 8yr deemed disposal, TCA 1997 Chapter 1A)
    - Exchange Traded Commodity -> ETC_COMMODITY (33% CGT)
    - Offshore distributing fund -> OFFSHORE_DISTRIBUTING (40% tax, TCA 1997 Part 27)
    - Direct equity / crypto / bond -> CGT_STANDARD (33% CGT, €1,270 exemption, TCA 1997 Section 580)

    Args:
        is_ucits: True if asset is a UCITS ETF or collective investment scheme.
        is_etc: True if asset is an Exchange Traded Commodity (e.g. physical gold ETC).
        is_offshore_distributing: True if asset is a qualifying offshore distributing fund.
        is_direct_equity_or_crypto: True if asset is confirmed direct stock, share, bond, or crypto asset.

    Returns:
        The matched IrishTaxRegime enum value.

    Raises:
        ValueError: If asset traits are unconfirmed or ambiguous.
    """
    if is_ucits:
        return IrishTaxRegime.EXIT_TAX
    if is_etc:
        return IrishTaxRegime.ETC_COMMODITY
    if is_offshore_distributing:
        return IrishTaxRegime.OFFSHORE_DISTRIBUTING
    if is_direct_equity_or_crypto:
        return IrishTaxRegime.CGT_STANDARD

    raise ValueError(
        "Cannot infer Irish tax regime: asset traits are unconfirmed or ambiguous. "
        "Explicit asset classification required to prevent favorable tax miscalculation."
    )


# ---------------------------------------------------------------------------
# SQLModel table definitions
# ---------------------------------------------------------------------------


class TaxpayerProfile(SQLModel, table=True):
    """Fiscal residency and domicile status for a specific tax year.

    Domicile vs Residence Distinction:
    - Domicile: Country of permanent home / legal origin (e.g. 'IT').
    - Residence: Physical presence in Ireland (>183 days in tax year).
    - Irish-domiciled residents are taxed on worldwide income/gains.
    - Non-domiciled Irish residents are taxed on foreign gains under
      the remittance basis only (gains taxed if remitted to Ireland).

    `is_domiciled_in_ireland` reflects whether `domicile_country == 'IE'`.
    """

    __tablename__ = "taxpayer_profile"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    tax_year: int = Field(..., index=True)
    fiscal_residence_country: str  # ISO 3166-1 alpha-2, e.g. 'IE', 'IT'
    domicile_country: str  # ISO 3166-1 alpha-2, e.g. 'IE', 'IT'
    residency_type: str = Field(..., description="Irish tax residency classification string.")
    marginal_tax_rate: Decimal = Field(..., description="Taxpayer's marginal income tax rate (e.g. 0.40 or 0.52).")
    notes: str | None = None

    @property
    def is_domiciled_in_ireland(self) -> bool:
        """Derived property: True if domicile country is Ireland ('IE')."""
        return self.domicile_country.upper() == "IE"


def parse_irish_tax_regime(raw: str | IrishTaxRegime) -> IrishTaxRegime:
    """Parse string or enum into IrishTaxRegime case-insensitively supporting values and member names.

    Args:
        raw: String value, member name, or IrishTaxRegime enum.

    Returns:
        The matched IrishTaxRegime enum value.

    Raises:
        ValueError: If raw value cannot be mapped to a known IrishTaxRegime.
    """
    if isinstance(raw, IrishTaxRegime):
        return raw
    normalized = raw.strip().lower()
    for regime in IrishTaxRegime:
        if normalized in (regime.value.lower(), regime.name.lower()):
            return regime
    raise ValueError(f"'{raw}' is not a valid IrishTaxRegime")


class AssetTaxClassificationDomain(BaseModel):
    """Domain model representing an asset's Irish tax regime classification."""

    isin: str = PydanticField(..., min_length=12, max_length=12, description="12-character ISIN code.")
    asset_name: str | None
    tax_regime: IrishTaxRegime
    domicile_country: str = PydanticField(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code."
    )
    is_ucits: bool
    is_etc: bool
    is_offshore_distributing: bool
    classification_source: str | None
    notes: str | None


class AssetTaxClassification(SQLModel, table=True):
    """Maps an ISIN to its applicable Irish tax regime.

    Populated manually by the user or enriched via OpenFIGI.
    The CGT engine hard-fails if a disposal targets an ISIN not present.
    """

    __tablename__ = "ireland_asset_tax_classification"  # type: ignore

    isin: str = Field(primary_key=True)
    asset_name: str | None = None
    tax_regime: str = Field(..., description="IrishTaxRegime enum value stored in database.")
    domicile_country: str | None = None  # ISO 3166-1 alpha-2
    is_ucits: bool = Field(default=False)
    classification_source: str | None = None  # 'manual' | 'openfigi_heuristic'
    notes: str | None = None
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_domain(self) -> AssetTaxClassificationDomain:
        """Convert database entity to strictly validated domain model."""
        regime = parse_irish_tax_regime(self.tax_regime)
        domicile = self.domicile_country

        if not domicile:
            raise ValueError("Domicile country is required for asset classification.")

        return AssetTaxClassificationDomain(
            isin=self.isin.upper().strip(),
            asset_name=self.asset_name,
            tax_regime=regime,
            domicile_country=domicile,
            is_ucits=self.is_ucits,
            is_etc=regime == IrishTaxRegime.ETC_COMMODITY,
            is_offshore_distributing=regime == IrishTaxRegime.OFFSHORE_DISTRIBUTING,
            classification_source=self.classification_source,
            notes=self.notes,
        )

    @classmethod
    def from_domain(cls, domain: AssetTaxClassificationDomain) -> "AssetTaxClassification":
        """Create database entity from strictly validated domain model."""
        return cls(
            isin=domain.isin.upper().strip(),
            asset_name=domain.asset_name,
            tax_regime=domain.tax_regime.value,
            domicile_country=domain.domicile_country.upper().strip(),
            is_ucits=domain.is_ucits,
            classification_source=domain.classification_source,
            notes=domain.notes,
        )


class RemittanceEvent(SQLModel, table=True):
    """Tracks remittance of disposal proceeds to Ireland.

    Used for non-domiciled residents under the remittance basis:
    foreign gains are only taxable if proceeds are brought into Ireland.
    """

    __tablename__ = "remittance_events"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    financial_record_id: int = Field(foreign_key="financial_records.id")
    remittance_date: datetime
    amount_eur: Decimal
    notes: str | None = None
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Disposal input models
# ---------------------------------------------------------------------------


class StrictDisposalInput(BaseModel):
    """Input for computing gains on an already-recorded sell transaction.

    Resolves the sell record from the database by primary key.
    """

    record_id: int


class SimulatedDisposalInput(BaseModel):
    """Input for computing hypothetical gains on a potential sell.

    All monetary values are in the specified currency; fx_rate converts
    to EUR for gain calculation.
    """

    isin: str
    quantity: Decimal
    disposal_date: date
    estimated_unit_price_eur: Decimal
    estimated_fees_eur: Decimal = Decimal("0")
    currency: str = "EUR"
    fx_rate: Decimal = PydanticField(
        default=Decimal("1.0"),
        description=(
            "Spot exchange rate converting transaction currency to EUR on transaction date "
            "(EUR_amount = foreign_amount * fx_rate). Defaults to 1.0 for EUR transactions."
        ),
    )


# Union type for disposal input
DisposalInput = StrictDisposalInput | SimulatedDisposalInput


# ---------------------------------------------------------------------------
# Computation result models
# ---------------------------------------------------------------------------


class LotMatch(BaseModel):
    """Result of matching a single FIFO buy lot against a disposal.

    Records the cost basis, proceeds, and gain/loss for the matched
    quantity, along with Section 581 bed-and-breakfasting flags.
    """

    source_record_id: int
    source_account_country: str
    buy_date: datetime
    matched_quantity: Decimal
    cost_basis_eur: Decimal = PydanticField(
        ...,
        description="Matched acquisition cost in EUR (includes pro-rata purchase fees).",
    )
    proceeds_eur: Decimal = PydanticField(
        ...,
        description="Matched disposal proceeds in EUR (net of pro-rata sale fees).",
    )
    gain_loss_eur: Decimal = PydanticField(
        ...,
        description="Realized capital gain (positive) or loss (negative) in EUR.",
    )
    is_section581_restricted: bool = PydanticField(
        default=False,
        description=(
            "Section 581 TCA 1997 Bed-and-Breakfasting rule: If shares of same class "
            "are repurchased within 4 weeks (28 days) after disposal at a loss, the loss "
            "is quarantined and cannot set off general capital gains."
        ),
    )
    section581_repurchase_record_id: int | None = PydanticField(
        default=None,
        description="Primary key ID of the repurchase transaction triggering Section 581 quarantine.",
    )


class DisposalResult(BaseModel):
    """Complete result of computing gains on a disposal event.

    Includes tax regime classification, lot matches, Section 581
    quarantine amounts, and Irish taxability determination.
    """

    disposal_record_id: int | None
    isin: str
    asset_name: str | None
    tax_regime: IrishTaxRegime
    disposal_date: datetime
    total_quantity: Decimal
    total_proceeds_eur: Decimal
    total_cost_basis_eur: Decimal
    gross_gain_loss_eur: Decimal
    unrestricted_gain_loss_eur: Decimal = PydanticField(
        ...,
        description=(
            "Net gain/loss excluding Section 581 quarantined losses. "
            "This is the net gain/loss available for offsetting against general capital gains "
            "or applying the €1,270 annual personal exemption."
        ),
    )
    section581_quarantined_loss_eur: Decimal = PydanticField(
        ...,
        description="Total capital loss quarantined under Section 581 bed-and-breakfasting rule.",
    )
    applicable_tax_rate: Decimal
    annual_exemption_applicable: bool
    loss_offset_allowed: bool
    deemed_disposal_applies: bool
    remittance_basis_applies: bool
    taxable_in_ireland: bool
    lot_matches: list[LotMatch]
    is_simulation: bool


class DeemedDisposalEvent(BaseModel):
    """A single lot that triggered the 8-year deemed disposal rule.

    Per Section 747D TCA 1997, UCITS ETF lots are deemed disposed
    on their 8th anniversary at fair market value.
    """

    source_record_id: int
    isin: str
    acquisition_date: datetime
    trigger_date: datetime
    quantity: Decimal
    original_cost_basis_eur: Decimal
    market_value_eur: Decimal = PydanticField(
        ...,
        description="Fair market value of units on the 8th anniversary trigger date (Section 747D TCA 1997).",
    )
    deemed_gain_eur: Decimal
    exit_tax_rate: Decimal
    exit_tax_due_eur: Decimal
    stepped_up_cost_per_unit_eur: Decimal = PydanticField(
        ...,
        description=(
            "Stepped-up base cost per unit in EUR following 8-year deemed disposal. "
            "Exit tax paid resets unit cost basis to fair market value on trigger date for future actual sales."
        ),
    )


class DeemedDisposalResult(BaseModel):
    """Aggregated result of scanning for 8-year deemed disposal triggers."""

    evaluation_date: date
    events: list[DeemedDisposalEvent]
    total_deemed_gain_eur: Decimal
    total_exit_tax_due_eur: Decimal


class ResolvedDisposalInput(BaseModel):
    """Normalized parameters resolved from raw or simulated disposal inputs."""

    isin: str
    sell_quantity: Decimal
    sell_unit_price_eur: Decimal
    sell_fees_eur: Decimal
    sell_date: datetime
    disposal_record_id: int | None = None
    asset_name: str | None = None
    is_simulation: bool = False


class AggregatedLotMatches(BaseModel):
    """Aggregated financial totals resulting from matching buy lots against a disposal."""

    total_proceeds_eur: Decimal
    total_cost_basis_eur: Decimal
    gross_gain_loss_eur: Decimal
    section581_quarantined_loss_eur: Decimal
    unrestricted_gain_loss_eur: Decimal


class RepurchaseAllocation(BaseModel):
    """Tracks a specific repurchase transaction available for Section 581(3) loss quarantine."""

    record_id: int
    remaining_qty: Decimal
