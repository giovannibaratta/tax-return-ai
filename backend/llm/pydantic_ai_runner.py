"""PydanticAI-based LLM Runner supporting text completions and schema-enforced structured outputs."""

import json
import logging
from typing import TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from backend.llm.runner import BaseLLMRunner

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _is_tool_calling_unsupported_error(exc: Exception) -> bool:
    """Check if an exception indicates the model or endpoint rejected forced tool calling."""
    msg = str(exc).lower()
    indicators = (
        "forced tool calling is not supported",
        "tool calling is not supported",
        "tools are not supported",
        "does not support tools",
        "does not support function",
        "tool_choice",
        "function calling is not supported",
    )
    return any(ind in msg for ind in indicators)


class PydanticAIRunner(BaseLLMRunner):
    """LLM Runner leveraging PydanticAI models with native structured output support and graceful fallback."""

    def __init__(self, model: Model) -> None:
        """Initialize PydanticAIRunner.

        Args:
            model: PydanticAI Model instance.
        """
        self._model = model
        self._model_name = model.model_name

    @property
    def model_name(self) -> str:
        """The name of the LLM model used by this runner."""
        return self._model_name

    @property
    def model(self) -> Model:
        """Underlying PydanticAI Model instance."""
        return self._model

    def complete(self, prompt: str, system_instruction: str = "") -> str:
        """Execute single-shot string completion via PydanticAI Agent.

        Args:
            prompt: User message / prompt.
            system_instruction: Optional system instruction string.

        Returns:
            Completed response string.
        """
        agent: Agent[None, str] = Agent(
            model=self._model,
            system_prompt=system_instruction or (),
        )
        result = agent.run_sync(prompt)
        return result.output

    def complete_structured(
        self,
        prompt: str,
        schema_cls: type[T],
        system_instruction: str = "",
        *,
        fallback_to_json: bool = False,
    ) -> T:
        """Execute single-shot schema-enforced structured extraction via PydanticAI Agent.

        Attempts native schema extraction first (via tool calling). If the underlying model/endpoint
        rejects forced tool calling and `fallback_to_json` is True, falls back to text completion
        with JSON schema instructions and manual Pydantic validation.

        Args:
            prompt: User extraction prompt.
            schema_cls: Target Pydantic model class for validated structured output.
            system_instruction: Optional system instructions.
            fallback_to_json: If True, falls back to prompt-guided JSON completion if tool calling fails.

        Returns:
            Validated instance of schema_cls.

        Raises:
            Exception: If extraction fails and fallback is disabled or unsuccessful.
        """
        try:
            agent: Agent[None, T] = Agent(
                model=self._model,
                output_type=schema_cls,
                system_prompt=system_instruction or (),
            )
            result = agent.run_sync(prompt)
            return result.output
        except Exception as exc:
            if not fallback_to_json or not _is_tool_calling_unsupported_error(exc):
                raise

            logger.warning(
                "Model '%s' does not support forced tool calling (%s). "
                "Falling back to prompt-guided JSON text completion with manual Pydantic validation.",
                self._model_name,
                exc,
            )
            return self._complete_structured_fallback(
                prompt=prompt,
                schema_cls=schema_cls,
                system_instruction=system_instruction,
            )

    def _complete_structured_fallback(
        self,
        prompt: str,
        schema_cls: type[T],
        system_instruction: str = "",
    ) -> T:
        """Fallback structured completion using pure text completion and Pydantic validation."""
        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
        fallback_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: You MUST respond ONLY with a single valid JSON object strictly matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any conversational preamble or markdown explanations outside the JSON."
        )

        raw_response = self.complete(fallback_prompt, system_instruction=system_instruction)
        cleaned_json = BaseLLMRunner.clean_json_response(raw_response)

        if not (cleaned_json.startswith("{") or cleaned_json.startswith("[")):
            start_idx = cleaned_json.find("{")
            end_idx = cleaned_json.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx : end_idx + 1].strip()

        return schema_cls.model_validate_json(cleaned_json)
