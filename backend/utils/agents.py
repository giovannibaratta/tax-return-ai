from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field

from backend.db_manager import DatabaseManager
from backend.llm.embedding_runner import BaseEmbeddingRunner


class BaseAccessedResource(BaseModel):
    """Base class for all accessed resources in tool traces."""

    access_type: Literal["read", "list"] = "read"


class DocumentChunkResource(BaseAccessedResource):
    """Accessed regulatory tax document chunk resource.

    Attributes:
        resource_type: Fixed discriminator tag 'document_chunk'.
        document_name: Name of source regulatory document.
        jurisdiction: Jurisdiction label ('italy', 'ireland'). Nullable if document is global.
        chunk_id: Primary key ID of tax document metadata chunk. Nullable when listing documents.
        page_number: Document page number. Nullable when chunk spans without precise page metadata.
        snippet: Excerpt snippet of chunk text. Nullable when only document references are listed.
    """

    resource_type: Literal["document_chunk"] = Field(default="document_chunk")
    document_name: str
    jurisdiction: str | None = None
    chunk_id: int | None = None
    page_number: int | None = None
    snippet: str | None = None


class FinancialRecordResource(BaseAccessedResource):
    """Accessed financial transaction record resource.

    Attributes:
        resource_type: Fixed discriminator tag 'financial_record'.
        record_id: Primary key ID of financial record.
    """

    resource_type: Literal["financial_record"] = Field(default="financial_record")
    record_id: int


class TaxpayerProfileResource(BaseAccessedResource):
    """Accessed taxpayer profile resource.

    Attributes:
        resource_type: Fixed discriminator tag 'taxpayer_profile'.
        tax_year: Tax year of queried profile.
        fiscal_residence_country: Country code of tax residence.
        domicile_country: Country code of domicile.
        residency_type: Fiscal classification.
    """

    resource_type: Literal["taxpayer_profile"] = Field(default="taxpayer_profile")
    tax_year: int
    fiscal_residence_country: str
    domicile_country: str
    residency_type: str


class DocumentPageResource(BaseModel):
    """Resource representing a full raw page retrieved from an indexed regulation document.

    Attributes:
        resource_type: Constant 'document_page'.
        document_name: Name of the regulatory source document (e.g. 'manual.pdf').
        jurisdiction: Jurisdiction label ('ireland', 'italy', etc.). Nullable if not specified.
        page_number: 1-indexed page number.
        total_pages: Total pages in the document.
        snippet: First 200 characters preview of the page text. Nullable if page has empty text.
    """

    resource_type: Literal["document_page"] = Field(default="document_page")
    document_name: str
    jurisdiction: str | None = None
    page_number: int
    total_pages: int
    snippet: str | None = None


# Discriminated Union supporting chunk resources, financial records, and document pages
AccessedResource = Annotated[
    DocumentChunkResource | FinancialRecordResource | DocumentPageResource | TaxpayerProfileResource,
    Discriminator("resource_type"),
]


class ToolCallInfo(BaseModel):
    """Information regarding a single tool invocation made by the chat agent.

    Attributes:
        tool_name: Name of invoked tool (e.g. 'query_tax_knowledge', 'calculate').
        args: Keyword arguments passed to tool.
        result_summary: High-level text summary of tool output.
        resources: List of specific resources/chunks retrieved or referenced by tool.
    """

    tool_name: str
    args: dict[str, object] = Field(default_factory=dict)
    result_summary: str = ""
    resources: list[AccessedResource] = Field(default_factory=lambda: list[AccessedResource]())


@dataclass
class SharedAgentDeps:
    """Shared dependencies provided to all tax and financial tool runs."""

    db: DatabaseManager
    embedding_runner: BaseEmbeddingRunner | None = None
    tool_traces: list[ToolCallInfo] = field(default_factory=lambda: list[ToolCallInfo]())
    on_progress: Callable[[str], None] | None = None
