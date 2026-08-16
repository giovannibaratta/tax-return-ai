# Generic Multi-Voter LLM Extraction and Consensus Engine

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from backend.llm.pydantic_ai_runner import PydanticAIRunner
from backend.llm.runner_factory import build_pydantic_ai_runner

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


_MIN_MAJORITY_COUNT: int = 2


class GenericConsensusResult(BaseModel, Generic[T]):
    """Result of a generic multi-voter extraction consensus run.

    Attributes:
        status: 'approved' if all voters agreed/reconciled, 'escalated' if mismatch, 'failed' on error.
        reconciled_output: Consolidated output model when status is 'approved' (or best majority candidate).
        voter_outputs: List of parsed models returned by each voter.
        discrepancies: List of human-readable differences detected across voters.
    """

    status: Literal["approved", "escalated", "failed"]
    reconciled_output: T | None = None
    voter_outputs: list[T]
    discrepancies: list[str]


def _reconcile_nested_models(field: str, values: list[object]) -> tuple[bool, object | None, list[str]]:
    """Reconcile nested Pydantic models recursively across voters."""
    nested_models = [v for v in values if isinstance(v, BaseModel)]
    if len(nested_models) != len(values):
        return False, None, [f"Field '{field}' has mixed BaseModel and non-BaseModel types across voters."]

    nested_reconciled, nested_disc = default_reconciler(nested_models)
    prefixed_disc = [f"Nested '{field}'.{disc}" for disc in nested_disc]
    if nested_reconciled is None:
        return False, None, prefixed_disc
    return True, nested_reconciled, prefixed_disc


def _reconcile_list_values(field: str, values: list[object]) -> tuple[bool, object | None, list[str]]:
    """Reconcile list values across voters using exact match or 2/3 majority."""
    list_dumps = [str(v) for v in values]
    if len(set(list_dumps)) == 1:
        return True, values[0], []

    for v in values:
        if values.count(v) >= _MIN_MAJORITY_COUNT:
            return True, v, [f"Field '{field}' list majority consensus: voters returned {values}"]

    return False, None, [f"Field '{field}' list has no majority consensus: {values}"]


def _reconcile_scalar_values(field: str, values: list[object]) -> tuple[bool, object | None, list[str]]:
    """Reconcile scalar values across voters using exact match or 2/3 majority."""
    unique_values: list[object] = []
    for v in values:
        if v not in unique_values:
            unique_values.append(v)

    if len(unique_values) == 1:
        return True, values[0], []

    for v in unique_values:
        if values.count(v) >= _MIN_MAJORITY_COUNT:
            return True, v, [f"Field '{field}' majority consensus ({v}): voters returned {values}"]

    return False, None, [f"Field '{field}' has no majority consensus: {values}"]


def _reconcile_field(field: str, values: list[object]) -> tuple[bool, object | None, list[str]]:
    """Dispatch reconciliation for a single field based on value type."""
    first_val = values[0]
    if isinstance(first_val, BaseModel):
        return _reconcile_nested_models(field, values)
    if isinstance(first_val, list):
        return _reconcile_list_values(field, values)
    return _reconcile_scalar_values(field, values)


def default_reconciler(outputs: list[T]) -> tuple[T | None, list[str]]:
    """Default reconciler comparing Pydantic models across voters for exact or majority consensus.

    Supports primitives, lists, and recursive nested Pydantic models.

    Args:
        outputs: List of validated model outputs from voters.

    Returns:
        Tuple of (reconciled model or None, list of discrepancy explanations).
    """
    if not outputs:
        return None, ["No voter outputs available to reconcile."]

    if len(outputs) == 1:
        return outputs[0], []

    # Fast path: check if all voters produced identical JSON dump
    dumps = [o.model_dump_json() for o in outputs]
    if len(set(dumps)) == 1:
        return outputs[0], []

    # Find differences across fields
    schema_cls = type(outputs[0])
    discrepancies: list[str] = []
    consensus_dict: dict[str, object] = {}

    for field in schema_cls.model_fields.keys():
        values = [getattr(o, field) for o in outputs]
        success, reconciled_val, field_disc = _reconcile_field(field, values)
        discrepancies.extend(field_disc)

        if not success:
            return None, discrepancies
        consensus_dict[field] = reconciled_val

    try:
        reconciled = schema_cls.model_validate(consensus_dict)
        return reconciled, discrepancies
    except Exception as err:
        discrepancies.append(f"Failed to instantiate reconciled model: {err}")
        return None, discrepancies


def run_multi_voter_consensus(
    prompt: str,
    system_prompt: str,
    schema_cls: type[T],
    runners: list[PydanticAIRunner] | None = None,
    reconciler: Callable[[list[T]], tuple[T | None, list[str]]] | None = None,
    *,
    max_attempts: int = 3,
) -> GenericConsensusResult[T]:
    """Execute multi-voter LLM extraction with schema validation and automated consensus reconciliation.

    Args:
        prompt: User or document extraction prompt.
        system_prompt: System prompt with domain and schema instructions.
        schema_cls: Expected Pydantic model class for extraction.
        runners: Optional list of PydanticAIRunner instances (defaults to VOTER_1, VOTER_2, VOTER_3).
        reconciler: Optional custom reconciliation function (defaults to default_reconciler).
        max_attempts: Number of query attempts per runner on schema validation failure.

    Returns:
        GenericConsensusResult containing status, reconciled output, all voter outputs, and discrepancies.
    """
    active_runners = runners
    if active_runners is None:
        active_runners = [
            build_pydantic_ai_runner("VOTER_1"),
            build_pydantic_ai_runner("VOTER_2"),
            build_pydantic_ai_runner("VOTER_3"),
        ]

    voter_outputs: list[T] = []
    reconcile_fn = reconciler or default_reconciler

    for idx, runner in enumerate(active_runners, start=1):
        parsed_output: T | None = None
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            logger.info("Querying Voter #%d (%s) - Attempt %d/%d...", idx, runner.model_name, attempt, max_attempts)
            try:
                parsed_output = runner.complete_structured(
                    prompt=prompt,
                    schema_cls=schema_cls,
                    system_instruction=system_prompt,
                )
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Voter #%d structured extraction failed on attempt %d: %s", idx, attempt, exc)

        if parsed_output is None:
            return GenericConsensusResult(
                status="failed",
                reconciled_output=None,
                voter_outputs=voter_outputs,
                discrepancies=[
                    f"Voter #{idx} ({runner.model_name}) failed after {max_attempts} attempts: {last_error}"
                ],
            )

        voter_outputs.append(parsed_output)

    reconciled, discrepancies = reconcile_fn(voter_outputs)
    status: Literal["approved", "escalated"] = "approved" if reconciled is not None else "escalated"

    return GenericConsensusResult(
        status=status,
        reconciled_output=reconciled,
        voter_outputs=voter_outputs,
        discrepancies=discrepancies,
    )


