"""Chat agent implementation backed by PydanticAI and financial/tax RAG tools."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeVar

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from backend.chat.models import AssistantChatMessage, ChatMessage, TokenUsageInfo
from backend.chat.summarizer import summarize_conversation_history
from backend.llm.interaction_logger import (
    log_request_start,
    log_response_finish,
)
from backend.llm.runner_factory import build_pydantic_model
from backend.tools.tax_tools import register_tax_tools
from backend.utils.agents import SharedAgentDeps, ToolCallInfo

logger = logging.getLogger(__name__)

CURRENT_DATE_STR = datetime.now(timezone.utc).strftime("%Y-%m")

CHAT_SYSTEM_PROMPT = (
    f"Current Date: {CURRENT_DATE_STR}\n"
    "You are a strict, factual tax compliance and financial information assistant.\n"
    "Your core directive is to provide 100% factually grounded answers strictly derived from "
    "retrieved source documents. False statements or hallucinated tax rules are strictly prohibited.\n\n"
    "STRICT OPERATIONAL WORKFLOW & RULES:\n"
    "1. **Check Taxpayer Profile First:** At the beginning of answering a tax query, fetch the taxpayer profile using `get_taxpayer_profile()` to establish the user's fiscal residence country (e.g., Ireland 'IE', Italy 'IT'), domicile country, and residency classification (e.g. resident_domiciled vs resident_non_domiciled).\n"
    "2. **Fetch Financial Records Before Knowledge Base:** If the user's message asks specific questions about financial transactions, holdings, assets, dividends, or trades, use `filter_financial_records` or `get_financial_record` to fetch actual financial records BEFORE checking the tax knowledge base.\n"
    "3. **Formulate Rich RAG Queries (STRICT MINIMUM 3 WORDS):** When invoking `query_tax_knowledge(query_text, ...)`, you MUST construct detailed, rich, multi-word descriptive queries (at least 3 words). Generic 1 or 2-word queries like 'tax', 'ETF', 'capital gains' will be REJECTED by system validation. Combine jurisdiction, asset class, tax rule or scenario (e.g. 'Ireland 8 year deemed disposal UCITS ETF exit tax', 'Italy 26% substitute tax capital gains dividend').\n"
    "4. **Zero Pre-Training Memory Reliance:** Do not rely on internal memory or general assumptions for tax rules, percentages, thresholds, or legal definitions. Every claim MUST be backed by retrieved tool evidence.\n"
    "5. **Strict Pushback on Missing Data:** If the requested information is not present in retrieved source documents, you MUST explicitly push back and state: 'I do not have sufficient information in the retrieved tax sources to answer this question.' Do NOT attempt to guess, extrapolate, or provide unverified advice.\n"
    "6. **Mandatory Citations & Quotes:** Always cite exact source documents (Document Name, Jurisdiction, Page Number, Chunk ID) for every fact or rule mentioned. Provide verbatim excerpts in an 'Appendix / Sources' section.\n"
    "7. **Deep Document Exploration (MANDATORY NEIGHBORS & PAGE READ):** When a RAG chunk from `query_tax_knowledge` is relevant, you MUST actively explore its context before concluding:\n"
    "   - Call `get_chunk_neighbors(chunk_id, window=2)` to retrieve adjacent paragraphs before and after the chunk.\n"
    "   - If a tax rule, table, or section spans across the whole page, or you want to have the full picture, call `read_doc_page(document_name, page_number, jurisdiction)` to read the full, un-truncated page text.\n"
    "8. **Verified Arithmetic:** For any tax liability or mathematical calculation, delegate computations to the `calculate(expression)` tool.\n\n"
    "Available tools:\n"
    "1. get_taxpayer_profile(tax_year): Retrieve taxpayer residency, domicile, and fiscal classification.\n"
    "2. filter_financial_records(asset_type, isin, quantity_over, quantity_less, purchase_date_start, purchase_date_end, logic): Filter financial transaction records by asset type, ISIN code, quantity thresholds, purchase dates, with AND/OR logic.\n"
    "3. get_financial_record(record_id): Fetch a specific transaction record by ID.\n"
    "4. query_tax_knowledge(query_text, limit, jurisdiction): Perform semantic search across indexed tax documents.\n"
    "5. list_documents(): View all available regulatory documents and manuals in database.\n"
    "6. get_chunk(chunk_id): Retrieve full text of a specific chunk by primary key ID.\n"
    "7. get_chunk_neighbors(chunk_id, window): Retrieve neighboring context chunks in the same document.\n"
    "8. read_doc_page(document_name, page_number, jurisdiction): Retrieve the complete raw text of an entire page.\n"
    "9. calculate(expression): Safely evaluate arithmetic expressions for precise tax math."
)


@dataclass(frozen=True)
class ChatOptions:
    """Configuration options for chat execution and history windowing.

    Attributes:
        max_history_tokens: Max estimated tokens for retained history.
        max_history_turns: Max exchange turns to retain.
        enable_summarization: Enable/disable summarization of older turns.
        request_limit: Initial limit on total tool/request turns.
        auto_approve_limit_extensions: Automatically approve extension when request limit is reached.
    """

    max_history_tokens: int = 20000
    max_history_turns: int = 30
    enable_summarization: bool = True
    request_limit: int = 50
    auto_approve_limit_extensions: bool = False

    @classmethod
    def from_env(cls) -> ChatOptions:
        """Create ChatOptions populated from environment variables with safe defaults.

        Returns:
            ChatOptions instance configured via environment variables.
        """
        return cls(
            max_history_tokens=int(os.getenv("CHAT_MAX_HISTORY_TOKENS") or "20000"),
            max_history_turns=int(os.getenv("CHAT_MAX_HISTORY_TURNS") or "30"),
            enable_summarization=os.getenv("CHAT_ENABLE_SUMMARIZATION", "true").lower() in ("true", "1", "yes"),
            request_limit=int(os.getenv("CHAT_REQUEST_LIMIT") or "50"),
            auto_approve_limit_extensions=os.getenv("CHAT_AUTO_APPROVE_LIMIT_EXTENSIONS", "false").lower()
            in ("true", "1", "yes"),
        )


@dataclass
class ChatDeps(SharedAgentDeps):
    """Dependencies container provided to chat agent tools during execution."""


def create_chat_agent(
    model: Model | str | None = None,
) -> Agent[ChatDeps, str]:
    """Create and configure a PydanticAI Agent with tax compliance tools registered.

    Args:
        model: Optional pre-configured Model instance or model string.

    Returns:
        Configured PydanticAI Agent instance for chat.
    """
    resolved_model: Model | str = model or build_pydantic_model("CHAT")

    agent = Agent(
        model=resolved_model,
        deps_type=ChatDeps,
        output_type=str,
        system_prompt=CHAT_SYSTEM_PROMPT,
    )

    register_tax_tools(agent)

    return agent


def extract_token_usage(usage_obj: RunUsage) -> TokenUsageInfo:
    """Extract TokenUsageInfo dataclass from PydanticAI RunUsage object.

    Args:
        usage_obj: PydanticAI RunUsage instance.

    Returns:
        TokenUsageInfo populated with request_tokens, response_tokens, total_tokens, cached_tokens.
    """
    req_tokens = usage_obj.input_tokens
    resp_tokens = usage_obj.output_tokens
    total_tokens = usage_obj.total_tokens
    cached_tokens = usage_obj.cache_read_tokens
    requests = usage_obj.requests

    return TokenUsageInfo(
        request_tokens=req_tokens,
        response_tokens=resp_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        requests=requests,
    )


def build_chat_history_prompt(
    past_messages: list[ChatMessage],
    options: ChatOptions | None = None,
    max_history_messages: int | None = None,
) -> str:
    """Build history prompt with configurable token/turn bounds and background summarization.

    Args:
        past_messages: Full list of past ChatMessage objects in session.
        options: Optional ChatOptions configuration dataclass.
        max_history_messages: Optional maximum turn override limit.

    Returns:
        Formatted history context string for LLM prompt.
    """
    if not past_messages:
        return ""

    opts = options or ChatOptions.from_env()
    max_tokens = opts.max_history_tokens
    max_turns = max_history_messages if max_history_messages is not None else opts.max_history_turns
    enable_summarization = opts.enable_summarization

    # Determine split index to keep recent_messages within BOTH max_turns and max_tokens bounds
    split_idx = len(past_messages)
    current_recent_tokens = 0
    retained_turns = 0

    for i in range(len(past_messages) - 1, -1, -1):
        msg_tokens = len(past_messages[i].content) // 4
        if retained_turns >= max_turns or (current_recent_tokens + msg_tokens > max_tokens and retained_turns > 0):
            break
        current_recent_tokens += msg_tokens
        retained_turns += 1
        split_idx = i

    older_messages = past_messages[:split_idx]
    recent_messages = past_messages[split_idx:]

    summary_prefix = ""
    if enable_summarization and older_messages:
        summary_text = summarize_conversation_history(older_messages)
        summary_prefix = f"Summary of Earlier Conversation:\n{summary_text}\n\n"

    history_lines: list[str] = []
    for msg in recent_messages:
        history_lines.append(f"{msg.role.capitalize()}: {msg.content}")

    history_str = "\n\n".join(history_lines)
    return f"{summary_prefix}Recent Conversation History:\n{history_str}"


DepsT = TypeVar("DepsT", bound=SharedAgentDeps)


# TODO: Why can't we extract the deps from the agent. I don't like to have to pass the agent and the deps around.
def run_chat_turn_sync(  # noqa: PLR0917
    agent: Agent[DepsT, str],
    deps: DepsT,
    prompt: str,
    past_messages: list[ChatMessage],
    options: ChatOptions | None = None,
    max_history_messages: int | None = None,
    request_limit: int | None = None,
    request_limit_callback: Callable[[int], bool] | None = None,
) -> tuple[ChatMessage, list[ToolCallInfo], TokenUsageInfo]:
    """Execute a single chat turn synchronously.

    Args:
        agent: PydanticAI Agent instance.
        deps: Dependencies container provided to agent and its tools.
        prompt: New user message string.
        past_messages: List of historical ChatMessages in session.
        options: Optional ChatOptions controlling limits and compaction behavior.
        max_history_messages: Optional maximum recent past messages override limit.
        request_limit: Optional initial request limit override.
        request_limit_callback: Optional approval callback (limit: int) -> bool invoked if limit reached.

    Returns:
        Tuple of (assistant ChatMessage response, list of ToolCallInfo executed, TokenUsageInfo metrics).
    """
    opts = options or ChatOptions.from_env()
    effective_request_limit = request_limit if request_limit is not None else opts.request_limit
    effective_deps = deps

    history_context = build_chat_history_prompt(
        past_messages,
        options=opts,
        max_history_messages=max_history_messages,
    )

    full_prompt = prompt
    if history_context:
        full_prompt = f"{history_context}\n\nCurrent User Question:\n{prompt}"

    model_name = str(getattr(agent.model, "model_name", "CHAT") if hasattr(agent, "model") else "CHAT")
    interaction_id = log_request_start(
        prompt=full_prompt,
        model_name=model_name,
        system_instruction=CHAT_SYSTEM_PROMPT,
        provider="pydantic-ai",
    )

    current_limit = effective_request_limit
    usage_info = TokenUsageInfo()

    while True:
        try:
            res = agent.run_sync(
                full_prompt,
                deps=effective_deps,
                usage_limits=UsageLimits(request_limit=current_limit),
            )
            output_content = res.output
            usage_info = extract_token_usage(res.usage)

            log_response_finish(
                interaction_id=interaction_id,
                response=output_content,
                status="COMPLETED",
                model_name=model_name,
            )
            break
        except UsageLimitExceeded as exc:
            logger.warning(
                "Execution limit reached at %d requests for agent turn: %s",
                current_limit,
                exc,
            )

            approved = opts.auto_approve_limit_extensions
            if not approved and request_limit_callback is not None:
                approved = request_limit_callback(current_limit)

            if approved:
                current_limit += effective_request_limit
                continue

            # User declined extension or no approval handler: return partial result safely
            output_content = (
                f"⚠️ Reached tool request limit ({current_limit} requests: {exc}).\n\n"
                f"Completed {len(effective_deps.tool_traces)} tool actions before reaching limit."
            )
            log_response_finish(
                interaction_id=interaction_id,
                response=output_content,
                status="CANCELLED_AT_LIMIT",
                model_name=model_name,
            )
            break
        except Exception as exc:
            logger.error("Agent execution failed: %s", exc)
            log_response_finish(
                interaction_id=interaction_id,
                response=str(exc),
                status="FAILED",
                model_name=model_name,
            )
            raise

    assistant_msg = AssistantChatMessage(
        content=output_content,
        tool_calls=effective_deps.tool_traces,
        usage=usage_info,
    )

    return assistant_msg, effective_deps.tool_traces, usage_info
