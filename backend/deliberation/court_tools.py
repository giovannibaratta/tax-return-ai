"""Tool functions registered on the courtroom deliberation agents.

Tools are registered via the ``register_tools`` helper which must be called
once after agent construction. This keeps tool registration decoupled from
agent declaration (``pydantic_agents.py``) so both can be imported independently
in tests.

Available tools:
- ``search_evidence``: Semantic KNN search over the regulatory document DB.
- ``get_chunk``: Retrieve a single chunk by its primary key.
- ``calculate``: Sandboxed arithmetic evaluator for verified tax math.
"""

# TODO: I don't fully get the implementation strategy of this file. For some tools, we access
# the ctx. For other we import them from tax_tools (seems better) and just wrapp for logging. Is there a better
# strategy that we can adopt to standardize this mechanism, preserve logging (maybe via a decorator or else),
# being able to declare them in one place and user them everywhere (E.g. MCP server)

import logging
from datetime import datetime, time
from decimal import Decimal
from typing import Any

from pydantic_ai import Agent, RunContext

from backend.deliberation.models import EvidenceChunk
from backend.deliberation.pydantic_agents import CourtDeps
from backend.utils.math_utils import evaluate_expression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_tools(*agents: Agent[CourtDeps, Any]) -> None:
    """Register all court tools on the provided agents.

    Call this once after constructing agents via ``make_plaintiff_agent``,
    ``make_defense_agent``, ``make_judge_agent``.

    Args:
        *agents: One or more ``Agent[CourtDeps, ...]`` instances to register tools on.
    """
    for agent in agents:
        _ = agent.tool(_search_evidence)
        _ = agent.tool(_get_chunk)
        _ = agent.tool(_calculate)
        _ = agent.tool(_get_financial_record)
        _ = agent.tool(_filter_financial_records)


async def _search_evidence(
    ctx: RunContext[CourtDeps],
    query: str,
    limit: int = 10,
) -> list[EvidenceChunk]:
    """Search the regulatory document database using semantic (KNN vector) search.

    Embeds ``query`` using the BGE-M3 model and retrieves the most relevant
    regulatory chunks for the active jurisdiction.

    Args:
        ctx: PydanticAI run context providing access to DB and embedding runner.
        query: A natural-language search query, e.g. ``"offshore fund exit tax Ireland"``.
        limit: Maximum number of chunks to return.

    Returns:
        A list of ``EvidenceChunk`` objects ordered by ascending cosine distance.
    """
    logger.info("Tool: search_evidence(query=%r, limit=%d)", query, limit)
    embedding = ctx.deps.embedding_runner.embed(query)
    return ctx.deps.db.semantic_search(
        query_embedding=embedding,
        limit=limit,
        jurisdiction=ctx.deps.jurisdiction,
    )


async def _get_chunk(
    ctx: RunContext[CourtDeps],
    chunk_id: int,
) -> EvidenceChunk | None:
    """Retrieve a single regulatory document chunk by its primary key.

    Use this when you already know the chunk ID (e.g. from a previous
    ``search_evidence`` call) and want to inspect its full content.

    Args:
        ctx: PydanticAI run context providing access to DB.
        chunk_id: The integer primary key of the chunk in ``tax_document_metadata``.

    Returns:
        The ``EvidenceChunk``, or ``None`` if not found.
    """
    logger.info("Tool: get_chunk(chunk_id=%d)", chunk_id)
    return ctx.deps.db.get_chunk_by_id(chunk_id)


def _calculate(
    _ctx: RunContext[CourtDeps],
    expression: str,
) -> Decimal:
    """Evaluate an arithmetic expression and return the verified numeric result.

    Use this tool for ALL tax math calculations. Never rely on your own mental
    arithmetic — always delegate numeric computation here to ensure accuracy.

    Supported operators: ``+``, ``-``, ``*``, ``/``, ``**`` (exponentiation).
    Numeric literals only — no variable names or function calls.

    Examples:
        - ``"3500 * 0.26"`` → 910.0
        - ``"12500 * 0.41 - 1270"`` → 3855.0
        - ``"(4000 - 1270) * 0.33"`` → 900.9

    Args:
        ctx: PydanticAI run context (not used directly, required by tool signature).
        expression: An arithmetic expression string.

    Returns:
        The computed float result.

    Raises:
        ValueError: If the expression is invalid or contains disallowed constructs.
    """
    logger.info("Tool: calculate(expression=%r)", expression)
    result = evaluate_expression(expression)
    logger.info("Tool: calculate result = %s", result)
    return result


async def _get_financial_record(
    ctx: RunContext[CourtDeps],
    record_id: int,
) -> dict[str, Any] | None:
    """Retrieve full details for a single financial transaction record by ID."""
    logger.info("Tool: get_financial_record(record_id=%d)", record_id)
    rec = ctx.deps.db.get_strict_financial_record(record_id)
    return rec.model_dump() if rec else None


async def _filter_financial_records(  # noqa: PLR0917
    ctx: RunContext[CourtDeps],
    asset_type: str | None = None,
    isin: str | None = None,
    quantity_over: float | None = None,
    quantity_less: float | None = None,
    purchase_date_start: str | None = None,
    purchase_date_end: str | None = None,
    logic: str = "AND",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Filter financial transaction records by asset type, ISIN code, quantity bounds, purchase dates, with AND/OR logic."""
    logger.info("Tool: filter_financial_records(asset_type=%r, isin=%r, logic=%r)", asset_type, isin, logic)

    start_dt = None
    if purchase_date_start:
        start_dt = (
            datetime.fromisoformat(purchase_date_start) if isinstance(purchase_date_start, str) else purchase_date_start
        )

    end_dt = None
    if purchase_date_end:
        end_dt = datetime.fromisoformat(purchase_date_end) if isinstance(purchase_date_end, str) else purchase_date_end
        if end_dt.time() == time.min:
            end_dt = datetime.combine(end_dt.date(), time.max)

    q_over = Decimal(str(quantity_over)) if quantity_over is not None else None
    q_less = Decimal(str(quantity_less)) if quantity_less is not None else None

    records = ctx.deps.db.filter_financial_records(
        asset_type=asset_type,
        isin=isin,
        quantity_over=q_over,
        quantity_less=q_less,
        purchase_date_start=start_dt,
        purchase_date_end=end_dt,
        logic=logic,
        account_country=ctx.deps.jurisdiction,
        limit=limit,
    )
    return [r.model_dump() for r in records]
