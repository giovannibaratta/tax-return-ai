"""Irish Tax Form (Form 11, Form 12 & Form CG1) data models for agent-driven reporting."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field

from backend.chat.models import ChatSession

FieldValueType = str | bool | Decimal
FieldStatus = Literal["computed_via_tool", "computed_via_rag", "user_override"]
TaxFormSection = Literal["capital_gains", "income", "tax_credits", "additional_fields"]


class FormField(BaseModel):
    """Represents a single field or logical block in a tax return form."""

    name: str = Field(description="Internal identifier for the field.")
    value: FieldValueType | None = Field(default=None, description="The computed or entered value.")
    status: FieldStatus = Field(
        description="Tracks how this field was populated.",
    )
    rationale: str | None = Field(
        default=None,
        description="The AI's explanation for this value. If computed_via_rag, must include expressions and data.",
    )


class FilingObligationDecision(BaseModel):
    """Assessment of the taxpayer's filing obligations for the target tax year."""

    required_form: Literal["undetermined", "form11", "form12", "cg1"] = Field(
        default="undetermined",
        description="Determined filing form (Form 11 for self-assessment, Form 12 for PAYE returns, Form CG1 for CGT-only).",
    )
    is_chargeable_person: bool | None = Field(
        default=None,
        description="Whether the individual meets criteria for a chargeable person under self-assessment.",
    )
    has_cgt_obligation: bool | None = Field(
        default=None,
        description="Whether the individual has chargeable capital gains or allowable losses in the tax year.",
    )
    rationale: str | None = Field(
        default=None,
        description="Legal and factual rationale justifying the selected filing form and obligation assessment.",
    )


class IrishUndeterminedState(BaseModel):
    """Initial state before taxpayer filing obligation has been determined."""

    form_type: Literal["undetermined"] = "undetermined"
    tax_year: int
    obligation_decision: FilingObligationDecision = Field(default_factory=FilingObligationDecision)
    additional_fields: dict[str, FormField] = Field(default_factory=dict)


class IrishForm11State(BaseModel):
    """Structured state of the Irish Form 11 tax return.

    Form 11 is the full self-assessment tax return covering both Income Tax and Capital Gains Tax.
    """

    form_type: Literal["form11"] = "form11"
    tax_year: int
    obligation_decision: FilingObligationDecision = Field(default_factory=FilingObligationDecision)

    # Strict deterministic sections (e.g., from CGT engine, PAYE calculator)
    capital_gains: dict[str, FormField] = Field(default_factory=dict)
    income: dict[str, FormField] = Field(default_factory=dict)

    # Free-form (open) sections for the rest of the form
    additional_fields: dict[str, FormField] = Field(default_factory=dict)


class IrishForm12State(BaseModel):
    """Structured state of the Irish Form 12 tax return.

    Form 12 is the tax return for PAYE employees / non-chargeable individuals with non-PAYE taxable
    income below the self-assessment threshold or claiming tax credits / reliefs.
    """

    form_type: Literal["form12"] = "form12"
    tax_year: int
    obligation_decision: FilingObligationDecision = Field(default_factory=FilingObligationDecision)

    # Sections for PAYE income, tax credits/reliefs, capital gains, and additional fields
    income: dict[str, FormField] = Field(default_factory=dict)
    tax_credits: dict[str, FormField] = Field(default_factory=dict)
    capital_gains: dict[str, FormField] = Field(default_factory=dict)
    additional_fields: dict[str, FormField] = Field(default_factory=dict)


class IrishCG1State(BaseModel):
    """Structured state of the Irish Form CG1 tax return.

    Form CG1 is the dedicated Capital Gains Tax return for individuals who do not need
    to file a full self-assessment Income Tax return (Form 11).
    """

    form_type: Literal["cg1"] = "cg1"
    tax_year: int
    obligation_decision: FilingObligationDecision = Field(default_factory=FilingObligationDecision)

    # Capital gains section
    capital_gains: dict[str, FormField] = Field(default_factory=dict)

    # Free-form (open) sections for additional CGT panels/reliefs
    additional_fields: dict[str, FormField] = Field(default_factory=dict)


IrishTaxFormState = Annotated[
    IrishUndeterminedState | IrishForm11State | IrishForm12State | IrishCG1State,
    Discriminator("form_type"),
]


class TaxFilingMetadata(BaseModel):
    """Structured metadata section stored within the tax filing session file."""

    tax_year: int = Field(description="Tax year covered by this filing return.")
    jurisdiction: str = Field(default="ireland", description="Applicable tax jurisdiction code.")


class IrishTaxFilingSession(ChatSession):
    """Chat session specialized for Irish tax return preparation.

    Attributes:
        metadata: Structured metadata section describing the tax return session.
        form_state: Active Irish tax form state (Undetermined, Form 11, Form 12, or Form CG1).
    """

    metadata: TaxFilingMetadata | None = Field(
        default=None,
        description="Structured metadata section (tax year, jurisdiction).",
    )
    form_state: IrishTaxFormState = Field(
        description="Active Irish tax form state (Undetermined, Form 11, Form 12, or Form CG1).",
    )

    def get_tax_year(self) -> int:
        """Return target tax year from metadata or active form state."""
        if self.metadata is not None:
            return self.metadata.tax_year
        return self.form_state.tax_year
