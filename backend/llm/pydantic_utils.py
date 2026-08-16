"""Shared PydanticAI helpers for caching and usage telemetry.

Provides two reusable utilities that any agent in the backend can import:

- ``get_cache_settings``: resolves the correct ``ModelSettings`` subclass to
  enable prompt caching for the active provider.
- ``log_agent_usage``: logs input/output/cache token counts at DEBUG level
  after an agent ``run_sync`` / ``run`` call.

Prompt caching is controlled globally via the ``PROMPT_CACHING_ENABLED``
environment variable (default: ``true``).
"""

import logging
import os

from pydantic_ai import AgentRunResult
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.settings import ModelSettings

from backend.llm.interaction_logger import log_interaction

# ---------------------------------------------------------------------------
# Caching flag
# ---------------------------------------------------------------------------

# Set PROMPT_CACHING_ENABLED=false to disable across the entire backend.
# Providers behave differently when caching is active:
#   - Anthropic (native AnthropicModel):  explicit cache breakpoints are
#     inserted via AnthropicModelSettings fields.
#   - OpenAI / OpenAI-compatible:         prefix caching is automatic for
#     eligible models (prefix > 1 024 tokens); no extra settings needed.
#   - Vertex / Gemini (GoogleModel):      context caching is provider-managed
#     and requires no extra model settings from the client.
PROMPT_CACHING_ENABLED: bool = os.environ.get("PROMPT_CACHING_ENABLED", "true").strip().lower() != "false"


# ---------------------------------------------------------------------------
# Cache settings helper
# ---------------------------------------------------------------------------


def get_cache_settings(model: Model | str | None) -> ModelSettings | None:
    """Return provider-specific ``ModelSettings`` that enable prompt caching.

    Pass the return value directly as ``model_settings`` in ``agent.run_sync()``
    or ``agent.run()``.

    Only Anthropic's native ``AnthropicModel`` requires an explicit opt-in
    via ``AnthropicModelSettings``; all other supported providers (OpenAI-
    compatible, Vertex/Gemini) apply caching automatically or at the
    infrastructure level, so ``None`` is returned for them.

    When ``PROMPT_CACHING_ENABLED`` is ``false`` this function always returns
    ``None``, effectively disabling the explicit opt-in for Anthropic and
    relying on each provider's default behavior for the others.

    Args:
        model: The PydanticAI ``Model`` instance bound to an agent. May also
            be a model-name string or ``None`` when the agent has no default
            model configured (PydanticAI allows late binding).

    Returns:
        An ``AnthropicModelSettings`` instance with all cache flags enabled,
        or ``None`` for any other provider / when caching is disabled.
    """
    if not PROMPT_CACHING_ENABLED:
        return None

    if isinstance(model, AnthropicModel):
        return AnthropicModelSettings(
            anthropic_cache=True,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
        )

    # OpenAI-compatible and Vertex auto-cache; no explicit settings required.
    return None


# ---------------------------------------------------------------------------
# Usage logging helper
# ---------------------------------------------------------------------------


def log_agent_usage(
    label: str,
    result: AgentRunResult[object],
    logger: logging.Logger,
    prompt: str | None = None,
) -> None:
    """Log token usage and record request/response interaction to disk.

    Reads PydanticAI's ``result.usage()`` to surface input, output, and
    cache metrics after each ``run_sync`` / ``run`` call.

    Args:
        label: Human-readable label for the turn (e.g. ``"Round 1 - Plaintiff"``).
        result: The ``AgentRunResult`` returned by ``agent.run_sync()`` or
            the resolved result of ``agent.run()``.
        logger: Logger to use.
        prompt: Optional prompt text passed to agent turn.
    """
    try:
        usage = result.usage
        cache_read = usage.cache_read_tokens
        cache_write = usage.cache_write_tokens
        hit_ratio = (cache_read / usage.input_tokens) if usage.input_tokens else 0.0
        logger.debug(
            "[%s] token usage — input: %s | output: %s" + " | cache_read: %s | cache_write: %s | hit_ratio: %.1f%%",
            label,
            usage.input_tokens,
            usage.output_tokens,
            cache_read,
            cache_write,
            hit_ratio * 100,
        )
    except Exception as exc:
        logger.debug("[%s] could not read usage metrics: %s", label, exc)

    try:
        model_name = label
        provider_name = "pydantic-ai"
        try:
            resp = result.response
            if resp.model_name:
                model_name = resp.model_name
            if resp.provider_name:
                provider_name = resp.provider_name
        except Exception:
            pass

        log_interaction(
            prompt=prompt or f"[{label}] Agent execution turn",
            response=str(result.output),
            model_name=model_name,
            provider=provider_name,
        )
    except Exception as exc:
        logger.debug("[%s] could not log interaction: %s", label, exc)
