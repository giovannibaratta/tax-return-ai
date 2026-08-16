"""PydanticAI-based LLM Runner supporting text completions and schema-enforced structured outputs."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from backend.llm.runner import BaseLLMRunner

T = TypeVar("T", bound=BaseModel)


class PydanticAIRunner(BaseLLMRunner):
    """LLM Runner leveraging PydanticAI models with native structured output support."""

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
    ) -> T:
        """Execute single-shot schema-enforced structured extraction via PydanticAI Agent.

        Args:
            prompt: User extraction prompt.
            schema_cls: Target Pydantic model class for validated structured output.
            system_instruction: Optional system instructions.

        Returns:
            Validated instance of schema_cls.
        """
        agent: Agent[None, T] = Agent(
            model=self._model,
            output_type=schema_cls,
            system_prompt=system_instruction or (),
        )
        result = agent.run_sync(prompt)
        return result.output
