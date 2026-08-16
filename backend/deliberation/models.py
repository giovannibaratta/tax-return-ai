from pydantic import BaseModel, Field, field_validator

from backend.domain_models import ConfidenceLevel, SourceType


class EvidenceChunk(BaseModel):
    """A single retrieved evidence chunk from the vector database.

    Attributes:
        id: Primary key of the chunk in the tax_document_metadata table.
        document_name: Name of the source PDF or markdown document.
        jurisdiction: Jurisdiction the document belongs to (e.g. 'italy', 'ireland', or None).
        page_number: Page number in the source document.
        text_content: The child chunk text used for retrieval (small, precise).
        chunk_index: Zero-based index of the chunk within the document.
        distance: Cosine distance from the query embedding (lower = more relevant).
            Present only when retrieved via semantic search; None for direct lookups.
        parent_chunk_id: Primary key of the parent chunk row, if this is a child chunk.
            None for parent rows or chunks ingested before parent-child support was added.
        parent_text_content: Full parent chunk text (large context window). When present,
            the deliberation layer should prefer this over text_content to give the LLM
            richer surrounding context.
        source_type: Provenance type ('regulation' or 'research').
        confidence_level: Authority confidence rating ('high', 'medium', 'low').
    """

    id: int
    document_name: str
    jurisdiction: str | None = None
    page_number: int
    text_content: str
    chunk_index: int
    distance: float | None = None
    parent_chunk_id: int | None = None
    parent_text_content: str | None = None
    source_type: SourceType
    confidence_level: ConfidenceLevel


class SourceDocument(BaseModel):
    """A single regulatory document citation from the judge's verdict.

    Attributes:
        regulatory_authority: Name of the issuing authority (e.g. 'Revenue Commissioners').
        document: Filename or title of the source document.
        page: Page number cited.
        section: Section or paragraph referenced within the page.
    """

    regulatory_authority: str
    document: str
    page: int | None = None
    section: str | None = None


class Traceability(BaseModel):
    """Typed evidence traceability block linking verdict claims to source documents.

    Attributes:
        source_documents: List of regulatory documents cited in the ruling.
        notes: Optional free-text comments about evidence quality or limitations.
    """

    source_documents: list[SourceDocument] = Field(default_factory=list)
    notes: str | None = None


class SourceConflict(BaseModel):
    """A detected inconsistency between a regulation source and a research source.

    Populated by the Judge agent when conflicting guidance is found.
    The system does NOT abort on conflict — this is an informational signal.
    """

    regulation_source: str
    regulation_claim: str
    research_source: str
    research_claim: str
    discrepancy_description: str


class TaxComputation(BaseModel):
    """Optional numeric computation block, present only when the verdict involves tax math.

    Attributes:
        calculated_field: Short label for what was computed (e.g. 'CGT liability 2025').
        values: Map of field labels to computed string values (e.g. {'cgt_liability': '€910'}).
        computation_formula: Human-readable formula showing how values were derived.
    """

    calculated_field: str
    values: dict[str, str]
    computation_formula: str


class CourtVerdict(BaseModel):
    """Structured output from the Judge agent at the end of a courtroom session.

    Replaces ``VerificationBlock``. Supports both computation scenarios (tax math)
    and consultation scenarios (compliance questions, regime eligibility, etc.).

    Attributes:
        ruling: The judge's final decision in plain language. Always present.
        computation: Tax math block, populated only when the scenario involves
            numeric calculations. None for pure consultation verdicts.
        traceability: Typed list of source regulatory documents cited as evidence.
        source_conflicts: List of detected discrepancies between regulation and research sources.
    """

    ruling: str
    computation: TaxComputation | None = None
    traceability: Traceability
    source_conflicts: list[SourceConflict] = Field(default_factory=list)


class DebateResult(BaseModel):
    """The full output of a completed courtroom debate session.

    Attributes:
        full_transcript: Complete debate transcript including the stable context prefix
            and all four rounds (plaintiff proposal, defense objection, plaintiff
            rebuttal, judge ruling).
        verdict: The raw text of the Judge's final ruling.
        court_verdict: Parsed structured CourtVerdict from the Judge's ruling.
    """

    full_transcript: str
    verdict: str
    court_verdict: CourtVerdict

    @property
    def has_source_conflicts(self) -> bool:
        """Return True if the Judge flagged any conflicting source guidance."""
        return len(self.court_verdict.source_conflicts) > 0


class TaxScenario(BaseModel):
    """Pydantic model representing a taxpayer case scenario file.

    Attributes:
        name: Short descriptive name for the case.
        description: Full factual narrative of the taxpayer's situation.
        jurisdiction: Active tax jurisdiction (must be 'italy' or 'ireland').
    """

    name: str = Field(default="Tax Court Case")
    description: str
    jurisdiction: str

    @field_validator("jurisdiction")
    @classmethod
    def validate_jurisdiction(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("italy", "ireland"):
            raise ValueError("Jurisdiction must be either 'italy' or 'ireland'")
        return clean
