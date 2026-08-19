"""Tests for Irish tax filing agent tools, normalization, and form state transitions."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.chat.tax_filing_agent import (
    TaxFilingDeps,
    clear_form_field,
    create_tax_filing_agent,
    set_filing_form,
    update_form_field,
)
from backend.db_manager import DatabaseManager, MemoryDb
from src.jurisdiction.ireland.tax_form_models import (
    FilingObligationDecision,
    IrishCG1State,
    IrishForm11State,
    IrishForm12State,
    IrishUndeterminedState,
    normalize_field_status,
    normalize_form_section,
    normalize_form_type,
)


def test_normalize_form_type() -> None:
    # Given: various string formats of Irish tax forms
    # When: normalized
    # Then: canonical form type identifier returned
    assert normalize_form_type("Form 11") == "form11"
    assert normalize_form_type("form 11") == "form11"
    assert normalize_form_type("Form-11") == "form11"
    assert normalize_form_type("form_11") == "form11"
    assert normalize_form_type("form11") == "form11"

    assert normalize_form_type("Form 12") == "form12"
    assert normalize_form_type("form 12") == "form12"
    assert normalize_form_type("form12") == "form12"

    assert normalize_form_type("Form CG1") == "cg1"
    assert normalize_form_type("CG1") == "cg1"
    assert normalize_form_type("form-cg1") == "cg1"
    assert normalize_form_type("cg1") == "cg1"

    assert normalize_form_type("undetermined") == "undetermined"
    assert normalize_form_type("unknown") == "undetermined"

    # Verify invalid form raises ValueError
    with pytest.raises(ValueError, match="Invalid tax form type"):
        normalize_form_type("11")

    with pytest.raises(ValueError, match="Invalid tax form type"):
        normalize_form_type("schedule_d")


def test_normalize_form_section_and_status() -> None:
    # Given: section and status string representations
    # When: normalized
    # Then: canonical section and status returned
    assert normalize_form_section("Capital Gains") == "capital_gains"
    assert normalize_form_section("capital_gains") == "capital_gains"
    assert normalize_form_section("income") == "income"
    assert normalize_form_section("Tax Credits") == "tax_credits"
    assert normalize_form_section("Additional Fields") == "additional_fields"

    # Verify invalid section raises ValueError
    with pytest.raises(ValueError, match="Invalid tax form section"):
        normalize_form_section("cgt")

    with pytest.raises(ValueError, match="Invalid tax form section"):
        normalize_form_section("paye")

    assert normalize_field_status("computed_via_tool") == "computed_via_tool"
    assert normalize_field_status("tool") == "computed_via_tool"
    assert normalize_field_status("computed_via_rag") == "computed_via_rag"
    assert normalize_field_status("rag") == "computed_via_rag"
    assert normalize_field_status("user_override") == "user_override"

    with pytest.raises(ValueError, match="Invalid field status"):
        normalize_field_status("random_method")


def test_set_filing_form_with_human_readable_name() -> None:
    # Given: An undetermined Irish tax form state in TaxFilingDeps
    db = DatabaseManager(MemoryDb())
    initial_state = IrishUndeterminedState(tax_year=2025)
    deps = TaxFilingDeps(db=db, form_state=initial_state)

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt="test",
        retry=0,
        tool_name="set_filing_form",
    )

    # When: set_filing_form is invoked with "Form 11" string (as LLMs often generate)
    res = asyncio.run(
        set_filing_form(
            ctx=ctx,
            form_type="Form 11",  # pyright: ignore[reportArgumentType]
            rationale="Chargeable person with non-PAYE trade income.",
            is_chargeable_person=True,
            has_cgt_obligation=True,
        )
    )

    # Then: form_state transitions to IrishForm11State successfully
    assert "FORM11" in res
    assert isinstance(deps.form_state, IrishForm11State)
    assert deps.form_state.tax_year == 2025
    assert deps.form_state.obligation_decision.required_form == "form11"
    assert deps.form_state.obligation_decision.is_chargeable_person is True


def test_set_filing_form_cg1_and_form12() -> None:
    # Given: TaxFilingDeps with initial Form 11 state
    db = DatabaseManager(MemoryDb())
    initial_state = IrishForm11State(tax_year=2025)
    deps = TaxFilingDeps(db=db, form_state=initial_state)

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt="test",
        retry=0,
        tool_name="set_filing_form",
    )

    # When: switching to Form CG1
    res_cg1 = asyncio.run(
        set_filing_form(
            ctx=ctx,
            form_type="Form CG1",  # pyright: ignore[reportArgumentType]
            rationale="PAYE employee with capital gains only.",
            is_chargeable_person=False,
            has_cgt_obligation=True,
        )
    )

    # Then: transitions to IrishCG1State
    assert "CG1" in res_cg1
    assert isinstance(deps.form_state, IrishCG1State)

    # When: switching to Form 12
    res_form12 = asyncio.run(
        set_filing_form(
            ctx=ctx,
            form_type="Form 12",  # pyright: ignore[reportArgumentType]
            rationale="PAYE employee claiming health expenses.",
            is_chargeable_person=False,
            has_cgt_obligation=False,
        )
    )

    # Then: transitions to IrishForm12State
    assert "FORM12" in res_form12
    assert isinstance(deps.form_state, IrishForm12State)


def test_update_and_clear_form_field() -> None:
    # Given: TaxFilingDeps with active Form 11 state
    db = DatabaseManager(MemoryDb())
    initial_state = IrishForm11State(tax_year=2025)
    deps = TaxFilingDeps(db=db, form_state=initial_state)

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt="test",
        retry=0,
        tool_name="update_form_field",
    )

    # When: updating capital gains field
    msg = asyncio.run(
        update_form_field(
            ctx=ctx,
            section="Capital Gains",  # pyright: ignore[reportArgumentType]
            field_name="Net_Chargeable_Gain",
            value=Decimal("4500.00"),
            rationale="Proceeds minus allowable cost minus annual exemption.",
            method="tool",  # pyright: ignore[reportArgumentType]
        )
    )

    # Then: field is stored in capital_gains dict
    assert "Successfully updated" in msg
    assert isinstance(deps.form_state, IrishForm11State)
    assert "Net_Chargeable_Gain" in deps.form_state.capital_gains
    field = deps.form_state.capital_gains["Net_Chargeable_Gain"]
    assert field.value == Decimal("4500.00")
    assert field.status == "computed_via_tool"

    # When: clearing field
    clear_msg = asyncio.run(
        clear_form_field(
            ctx=ctx,
            section="Capital Gains",  # pyright: ignore[reportArgumentType]
            field_name="Net_Chargeable_Gain",
        )
    )

    # Then: field is removed
    assert "Successfully cleared" in clear_msg
    assert isinstance(deps.form_state, IrishForm11State)
    assert "Net_Chargeable_Gain" not in deps.form_state.capital_gains


def test_filing_obligation_decision_pydantic_validation() -> None:
    # Given: raw dictionary with human-formatted form names
    # When: parsed into FilingObligationDecision
    # Then: values are normalized to canonical literals
    dec1 = FilingObligationDecision.model_validate({"required_form": "Form 11"})
    assert dec1.required_form == "form11"

    dec2 = FilingObligationDecision.model_validate({"required_form": "Form 12"})
    assert dec2.required_form == "form12"

    dec3 = FilingObligationDecision.model_validate({"required_form": "Form CG1"})
    assert dec3.required_form == "cg1"


def test_create_tax_filing_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: LLM provider environment variables
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("DEFAULT_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("DEFAULT_LLM_AUTH_TYPE", "API_KEY")
    monkeypatch.setenv("DEFAULT_LLM_API_KEY", "test-api-key")

    # When: creating agent
    agent = create_tax_filing_agent()

    # Then: agent is created with specialized tools registered
    toolset = getattr(agent, "_function_toolset", None)
    tool_names: set[str] = set(getattr(toolset, "tools", {}).keys())
    assert "get_form_state" in tool_names
    assert "set_filing_form" in tool_names
    assert "update_form_field" in tool_names
    assert "clear_form_field" in tool_names
    assert "run_cgt_computation" in tool_names
    assert "run_pension_calculator" in tool_names
