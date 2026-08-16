"""Tests for PydanticAIRunner text completion and structured extraction."""

from decimal import Decimal

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from backend.llm.pydantic_ai_runner import PydanticAIRunner


class SampleExtractionSchema(BaseModel):
    name: str
    amount: Decimal


def test_pydantic_ai_runner_complete() -> None:
    # Given: A TestModel configured with custom response
    model = TestModel(custom_output_text="Test completed successfully")
    runner = PydanticAIRunner(model=model)

    # When: Calling complete
    result = runner.complete("Hello world")

    # Then: Returns string output
    assert result == "Test completed successfully"
    assert runner.model_name == "test"


def test_pydantic_ai_runner_complete_structured() -> None:
    # Given: A TestModel configured with custom output args for the schema
    model = TestModel(custom_output_args={"name": "Alpha Corp", "amount": "1250.50"})
    runner = PydanticAIRunner(model=model)

    # When: Calling complete_structured
    result = runner.complete_structured("Extract entity", schema_cls=SampleExtractionSchema)

    # Then: Returns strongly-typed validated model
    assert isinstance(result, SampleExtractionSchema)
    assert result.name == "Alpha Corp"
    assert result.amount == Decimal("1250.50")
