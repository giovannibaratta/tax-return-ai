"""Module for summarizing historical chat turns to maintain context cache stability."""

from __future__ import annotations

from backend.chat.models import ChatMessage
from backend.llm.runner_factory import build_runner

SUMMARIZER_PROMPT = (
    "You are a strict, objective conversation history summarizer for a tax return assistant.\n"
    "Your job is to condense a sequence of older chat turns into a compact, highly structured summary block.\n"
    "PRESERVE ALL:\n"
    "1. Taxpayer profile facts (fiscal residence, domicile, tax years, residency classification).\n"
    "2. Financial assets mentioned (symbols, ISINs, quantities, dates, actions, transaction IDs).\n"
    "3. Specific tax laws, articles, manual sections, exit tax rules, or CGT percentages cited.\n"
    "4. Pending questions or unresolved user requests.\n\n"
    "Format the summary clearly as markdown bullet points. Do NOT include pleasantries."
)


def summarize_conversation_history(
    messages: list[ChatMessage],
    provider_prefix: str = "CHAT_SUMMARIZER",
) -> str:
    """Summarize a sequence of older chat turns into a compact summary string.

    Args:
        messages: List of ChatMessage instances to condense.
        provider_prefix: Config prefix for LLM runner (defaults to CHAT_SUMMARIZER, falls back to DEFAULT).

    Returns:
        Structured summary markdown string.
    """
    if not messages:
        return ""

    formatted_turns: list[str] = []
    for msg in messages:
        role = msg.role.capitalize()
        formatted_turns.append(f"{role}: {msg.content}")

    conversation_text = "\n\n".join(formatted_turns)

    runner = build_runner(provider_prefix)
    summary_result = runner.complete(
        prompt=f"Please summarize the following older conversation turns:\n\n{conversation_text}",
        system_instruction=SUMMARIZER_PROMPT,
    )
    return summary_result.strip()
