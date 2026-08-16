"""Standalone reusable tax tool actions and PydanticAI agent tool registration helpers.

Provides decoupled tool actions for:
- Semantic tax RAG search (`query_tax_knowledge_action`)
- Document listing (`list_documents_action`)
- Single chunk retrieval (`get_chunk_action`)
- Neighboring chunk exploration (`get_chunk_neighbors_action`)
- Safe arithmetic evaluation (`calculate_action`)
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, ParamSpec, TypeVar, cast

from pydantic_ai import Agent, RunContext

from backend.deliberation.models import EvidenceChunk
from backend.domain_models import BaseStrictRecord, DocumentMetadata, DocumentPageInfo, StrictTaxIncomeRecord
from backend.services.tax_services import (
    calculate_action,
    filter_financial_records_action,
    get_chunk_action,
    get_chunk_neighbors_action,
    get_financial_record_action,
    get_tax_income_records_action,
    get_taxpayer_profile_action,
    list_documents_action,
    query_tax_knowledge_action,
    read_doc_page_action,
)
from backend.utils.agents import (
    AccessedResource,
    DocumentChunkResource,
    DocumentPageResource,
    FinancialRecordResource,
    SharedAgentDeps,
    TaxpayerProfileResource,
    ToolCallInfo,
)
from src.jurisdiction.ireland.cgt_models import TaxpayerProfile

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def trace_tool(
    tool_name: str,
    summary_fn: Callable[[Any], str],
    resources_fn: Callable[[Any], list[AccessedResource]] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator to automatically trace PydanticAI tool executions."""

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            trace_args = dict(bound.arguments)

            ctx: RunContext[SharedAgentDeps] | None = trace_args.pop("ctx", None)

            if ctx is None and args and isinstance(args[0], RunContext):
                ctx = cast(RunContext[SharedAgentDeps], args[0])
                first_param_name = next(iter(sig.parameters.keys()))
                trace_args.pop(first_param_name, None)

            if ctx is None:
                raise RuntimeError("RunContext not found in tool arguments")

            logger.info("Tool: %s(%s)", tool_name, ", ".join(f"{k}={v!r}" for k, v in trace_args.items()))
            if ctx.deps.on_progress:
                arg_str = ", ".join(f"{k}={v!r}" for k, v in trace_args.items())
                ctx.deps.on_progress(f"Executing {tool_name}({arg_str})")

            try:
                res = await func(*args, **kwargs)
                summary = summary_fn(res)
                resources = resources_fn(res) if resources_fn else []
                trace = ToolCallInfo(
                    tool_name=tool_name,
                    args=trace_args,
                    result_summary=summary,
                    resources=resources,
                )
                ctx.deps.tool_traces.append(trace)
                return res
            except Exception as err:
                trace = ToolCallInfo(
                    tool_name=tool_name,
                    args=trace_args,
                    result_summary=f"Failed: {err}",
                    resources=[],
                )
                ctx.deps.tool_traces.append(trace)
                raise ValueError(str(err)) from err

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# PydanticAI Agent Tool Registration Helper
# ---------------------------------------------------------------------------


DepsT = TypeVar("DepsT", bound=SharedAgentDeps)
ResultT = TypeVar("ResultT")


def register_tax_tools(*agents: Agent[DepsT, ResultT]) -> None:
    """Register reusable tax tools on the provided PydanticAI agents.

    Args:
        *agents: One or more PydanticAI Agent instances whose dependencies provide ``SharedAgentDeps``.
    """
    for agent in agents:
        _ = agent.tool(get_taxpayer_profile)
        _ = agent.tool(query_tax_knowledge)
        _ = agent.tool(list_documents)
        _ = agent.tool(get_chunk)
        _ = agent.tool(get_chunk_neighbors)
        _ = agent.tool(read_doc_page)
        _ = agent.tool(calculate)
        _ = agent.tool(get_financial_record)
        _ = agent.tool(filter_financial_records)
        _ = agent.tool(get_tax_income_records)


@trace_tool(
    tool_name="get_taxpayer_profile",
    summary_fn=lambda res: f"Retrieved {len(res)} taxpayer profile record(s)",
    resources_fn=lambda res: [
        TaxpayerProfileResource(
            tax_year=p.tax_year,
            fiscal_residence_country=p.fiscal_residence_country,
            domicile_country=p.domicile_country,
            residency_type=p.residency_type,
        )
        for p in res
    ],
)
async def get_taxpayer_profile(
    ctx: RunContext[SharedAgentDeps],
    tax_year: int | None = None,
) -> list[TaxpayerProfile]:
    """Retrieve the taxpayer's fiscal residency, domicile, and tax classification status.

    Use this tool at the beginning of a turn to determine the taxpayer's fiscal residence country
    (e.g., Ireland 'IE', Italy 'IT'), domicile country, and residency type (e.g. resident_domiciled,
    resident_non_domiciled) to provide precise, jurisdictionally accurate answers.
    """
    return get_taxpayer_profile_action(db=ctx.deps.db, tax_year=tax_year)


@trace_tool(
    tool_name="query_tax_knowledge",
    summary_fn=lambda res: f"Retrieved {len(res)} matching chunks",
    resources_fn=lambda res: [
        DocumentChunkResource(
            document_name=item.document_name,
            jurisdiction=item.jurisdiction,
            chunk_id=item.id,
            page_number=item.page_number,
            snippet=item.text_content,
        )
        for item in res
    ],
)
async def query_tax_knowledge(
    ctx: RunContext[SharedAgentDeps],
    query_text: str,
    limit: int = 5,
    jurisdiction: str | None = None,
) -> list[EvidenceChunk]:
    """Perform semantic search against indexed tax guidelines and regulatory documents.

    IMPORTANT: Always pass rich, detailed, multi-word descriptive queries (e.g.
    'Italy substitute tax rate on financial asset capital gains 26%', 'Irish CGT personal
    exemption threshold and disposal rules'). NEVER use single-word queries like 'tax' or 'capital'.
    """
    words = [w for w in query_text.strip().split() if w]
    if len(words) < 3:
        err_msg = (
            f"Query '{query_text}' rejected: Search query must contain at least 3 descriptive words "
            f"(e.g., 'Ireland 8 year deemed disposal UCITS ETF exit tax', 'Italy 26% substitute tax capital gains'). "
            f"Do not pass generic 1 or 2-word queries like '{query_text}'."
        )
        logger.warning(err_msg)
        raise ValueError(err_msg)

    return query_tax_knowledge_action(
        db=ctx.deps.db,
        embedding_runner=ctx.deps.embedding_runner,
        query_text=query_text,
        limit=limit,
        jurisdiction=jurisdiction,
    )


@trace_tool(
    tool_name="list_documents",
    summary_fn=lambda res: f"Found {len(res)} indexed documents",
    resources_fn=lambda res: [
        DocumentChunkResource(
            document_name=doc.document_name,
            jurisdiction=doc.jurisdiction,
            access_type="list",
        )
        for doc in res
    ],
)
async def list_documents(ctx: RunContext[SharedAgentDeps]) -> list[DocumentMetadata]:
    """List all tax regulatory documents and manuals indexed in database."""
    return list_documents_action(db=ctx.deps.db)


@trace_tool(
    tool_name="get_chunk",
    summary_fn=lambda res: f"Retrieved chunk {res.id}" if res else "Chunk not found",
    resources_fn=lambda res: (
        [
            DocumentChunkResource(
                document_name=res.document_name,
                jurisdiction=res.jurisdiction,
                chunk_id=res.id,
                page_number=res.page_number,
                snippet=res.text_content,
            )
        ]
        if res
        else []
    ),
)
async def get_chunk(ctx: RunContext[SharedAgentDeps], chunk_id: int) -> EvidenceChunk | None:
    """Retrieve full details for a single document chunk by ID."""
    return get_chunk_action(db=ctx.deps.db, chunk_id=chunk_id)


@trace_tool(
    tool_name="get_chunk_neighbors",
    summary_fn=lambda res: f"Retrieved {len(res)} neighbor chunks",
    resources_fn=lambda res: [
        DocumentChunkResource(
            document_name=chunk.document_name,
            jurisdiction=chunk.jurisdiction,
            chunk_id=chunk.id,
            page_number=chunk.page_number,
            snippet=chunk.text_content,
        )
        for chunk in res
    ],
)
async def get_chunk_neighbors(ctx: RunContext[SharedAgentDeps], chunk_id: int, window: int = 1) -> list[EvidenceChunk]:
    """Retrieve neighboring context chunks (before and after) in the same document."""
    return get_chunk_neighbors_action(db=ctx.deps.db, chunk_id=chunk_id, window=window)


@trace_tool(
    tool_name="read_doc_page",
    summary_fn=lambda res: (
        f"Retrieved page {res.page_number}/{res.total_pages} of {res.document_name}" if res else "Page not found"
    ),
    resources_fn=lambda res: (
        [
            DocumentPageResource(
                document_name=res.document_name,
                jurisdiction=res.jurisdiction,
                page_number=res.page_number,
                total_pages=res.total_pages,
                snippet=res.text_content,
            )
        ]
        if res
        else []
    ),
)
async def read_doc_page(
    ctx: RunContext[SharedAgentDeps],
    document_name: str,
    page_number: int,
    jurisdiction: str | None = None,
) -> DocumentPageInfo | None:
    """Retrieve the complete raw text of an entire page from an indexed regulatory document.

    Use this tool when RAG search chunks or chunk neighbors do not provide the full context of a
    page, or when a tax regulation table/rule spans across an entire page.
    """
    return read_doc_page_action(
        db=ctx.deps.db,
        document_name=document_name,
        page_number=page_number,
        jurisdiction=jurisdiction,
    )


@trace_tool(
    tool_name="calculate",
    summary_fn=lambda res: f"Result = {res}",
    resources_fn=lambda res: [],
)
async def calculate(ctx: RunContext[SharedAgentDeps], expression: str) -> Decimal:
    """Safely evaluate an arithmetic expression."""
    return calculate_action(expression)


@trace_tool(
    tool_name="get_financial_record",
    summary_fn=lambda res: f"Retrieved financial record {res.id}" if res else "Record not found",
    resources_fn=lambda res: (
        [
            FinancialRecordResource(
                record_id=res.id,
            )
        ]
        if res
        else []
    ),
)
async def get_financial_record(ctx: RunContext[SharedAgentDeps], record_id: int) -> BaseStrictRecord | None:
    """Retrieve full details for a single financial transaction record by ID."""
    return get_financial_record_action(db=ctx.deps.db, record_id=record_id)


@trace_tool(
    tool_name="filter_financial_records",
    summary_fn=lambda res: f"Found {len(res)} matching financial records",
    resources_fn=lambda res: [
        FinancialRecordResource(
            record_id=r.id,
        )
        for r in res
    ],
)
async def filter_financial_records(  # noqa: PLR0917
    ctx: RunContext[SharedAgentDeps],
    asset_type: Literal["stock", "etf", "cash", "tax_payment", "salary", "pension"] | None = None,
    action: Literal["buy", "sell", "dividend", "tax_payment", "salary_payout", "corporate_action"] | None = None,
    tax_year: int | None = None,
    isin: str | None = None,
    quantity_over: Decimal | None = None,
    quantity_less: Decimal | None = None,
    purchase_date_start: datetime | None = None,
    purchase_date_end: datetime | None = None,
    logic: Literal["AND", "OR"] = "AND",
    account_country: str | None = None,
    limit: int = 100,
) -> list[BaseStrictRecord]:
    """Filter financial records by asset_type, action, tax_year, ISIN, quantity, and purchase date.

    Args:
        ctx: RunContext containing SharedAgentDeps.
        asset_type: Optional asset class ('stock', 'etf', 'cash', 'tax_payment', 'salary', 'pension').
        action: Optional transaction action ('buy', 'sell', 'dividend', 'tax_payment', 'salary_payout', 'corporate_action').
        tax_year: Optional tax year integer (e.g. 2025).
        isin: Optional ISIN identifier (e.g. 'IE00B4L5Y983').
        quantity_over: Optional quantity lower bound.
        quantity_less: Optional quantity upper bound.
        purchase_date_start: Optional start date/time.
        purchase_date_end: Optional end date/time.
        logic: Combination logic ('AND' or 'OR').
        account_country: Optional account country ('IE', 'IT').
        limit: Max records to return.
    """
    return filter_financial_records_action(
        db=ctx.deps.db,
        asset_type=asset_type,
        action=action,
        tax_year=tax_year,
        isin=isin,
        quantity_over=quantity_over,
        quantity_less=quantity_less,
        purchase_date_start=purchase_date_start,
        purchase_date_end=purchase_date_end,
        logic=logic,
        account_country=account_country,
        limit=limit,
    )


@trace_tool(
    tool_name="get_tax_income_records",
    summary_fn=lambda res: f"Retrieved {len(res)} verified tax income record(s)",
    resources_fn=lambda res: [],
)
async def get_tax_income_records(
    ctx: RunContext[SharedAgentDeps],
    tax_year: int | None = None,
    jurisdiction: str | None = None,
) -> list[StrictTaxIncomeRecord]:
    """Retrieve verified official tax income records (e.g. Irish Employment Detail Summary / EDS).

    Use this tool to obtain official PAYE employment income figures (gross pay, tax paid, USC, PRSI)
    for a given tax year and jurisdiction (e.g. 'ireland').

    Args:
        ctx: RunContext containing SharedAgentDeps.
        tax_year: Optional tax year integer (e.g. 2025).
        jurisdiction: Optional jurisdiction filter ('ireland', 'italy').

    Returns:
        List of StrictTaxIncomeRecord domain models.
    """
    return get_tax_income_records_action(
        db=ctx.deps.db,
        tax_year=tax_year,
        jurisdiction=jurisdiction,
    )

