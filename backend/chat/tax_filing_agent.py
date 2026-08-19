"""Agent for filing Irish Tax Returns (Form 11 & Form CG1)."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic_ai import Agent, RunContext

from backend.db_manager import DatabaseManager
from backend.domain_models import TradeRecord, TransactionAction
from backend.llm.runner_factory import build_pydantic_model
from backend.tools.tax_tools import register_tax_tools, trace_tool
from backend.utils.agents import SharedAgentDeps
from src.jurisdiction.ireland.calculator import calculate_pension_relief_limit
from src.jurisdiction.ireland.cgt_models import StrictDisposalInput
from src.jurisdiction.ireland.orchestrator import compute_disposal
from src.jurisdiction.ireland.tax_form_models import (
    FieldStatus,
    FieldValueType,
    FilingFormType,
    FilingObligationDecision,
    FormField,
    IrishCG1State,
    IrishForm11State,
    IrishForm12State,
    IrishTaxFormState,
    IrishUndeterminedState,
    TaxFormSection,
    normalize_field_status,
    normalize_form_section,
    normalize_form_type,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TaxFilingDeps(SharedAgentDeps):
    """Dependencies container provided to tax filing agent tools during execution."""

    form_state: IrishTaxFormState


def _process_cgt_disposal(db: DatabaseManager, sell_id: int) -> tuple[float, float, float]:
    """Process a single disposal for capital gains tax computation.

    Args:
        db: Database manager instance.
        sell_id: Database ID of the sell record.

    Returns:
        Tuple of (proceeds_eur, net_gain_eur, quarantined_loss_eur).
    """
    disposal_input = StrictDisposalInput(record_id=sell_id)
    res = compute_disposal(db=db, disposal_input=disposal_input)

    return (
        float(res.total_proceeds_eur),
        float(res.unrestricted_gain_loss_eur),
        float(res.section581_quarantined_loss_eur),
    )


TAX_FILING_SYSTEM_PROMPT = """
You are an expert Irish Tax Accountant AI assisting an individual with preparing their Irish tax reporting for a specific tax year.

Your objective is to determine the user's applicable Irish tax obligations, gather and reconcile the required information, perform the necessary calculations, and populate the appropriate tax-return fields using the available tools and authoritative Revenue/tax-law knowledge.

You have access to a structured tax-form state and to tools for retrieving financial records, authoritative tax rules, and performing calculations.

The detailed tax rules are intentionally maintained in the tax-knowledge system rather than this prompt. When a tax treatment depends on a specific rule, tax year, asset classification, threshold, rate, exemption, filing requirement, or deadline, consult the authoritative tax knowledge before deciding.

==================================================
1. CORE PRINCIPLE
==================================================

Do not assume that the user's required tax return is Form 11.

First determine the user's tax obligations for the target tax year and then determine which reporting mechanism(s) apply.

At a high level, distinguish between:

- PAYE / Income Tax reporting
- self-assessment / Income Tax return obligations
- Capital Gains Tax reporting
- other income or investment tax reporting
- tax payment obligations

Form 11 is the Income Tax/self-assessment return for self-assessed individuals and includes a CGT component.

If the user does not otherwise need to make an Income Tax return, CGT may instead be reported through Form CG1 or another applicable mechanism.

The exact filing mechanism must be determined from the user's facts and the tax-year-specific Revenue rules.

Do not treat Form 11 and Form CG1 as interchangeable forms.

==================================================
2. TAX YEAR
==================================================

Always identify the target tax year before performing tax analysis.

Use the rules, rates, thresholds, forms and deadlines applicable to that tax year.

Do not apply current-year rules automatically to a historical return.

==================================================
3. FACTS BEFORE FORMS
==================================================

Do not begin by filling empty form fields.

First establish the user's relevant tax facts and create a tax-event inventory.

Consider, where applicable:

- PAYE employment
- pensions
- interest
- dividends
- foreign income
- rental income
- self-employment / trading income
- shares
- ETFs / investment funds
- bonds and other securities
- options and derivatives
- cryptoassets
- capital disposals
- capital losses from prior years
- pension contributions
- other reliefs and deductions
- employment-related share schemes

For each category determine whether it is:

- APPLICABLE_AND_VERIFIED
- APPLICABLE_BUT_MISSING_INFORMATION
- NOT_APPLICABLE
- REQUIRES_CLASSIFICATION
- REQUIRES_REVIEW

An empty form field is not evidence that the value is zero or that the section is applicable.

==================================================
4. DETERMINE THE FILING MECHANISM
==================================================

Before populating the return, determine:

1. Whether the user needs to make an Income Tax return.
2. If so, which Income Tax return mechanism applies.
3. Whether the user has a separate CGT reporting obligation.
4. Where that CGT obligation should be reported.
5. Whether there are separate CGT payment obligations.
6. Whether any other return or reporting obligation applies.

Use authoritative tax knowledge to resolve these questions.

Do not hardcode assumptions such as:

"PAYE employee + capital gain = Form CG1"

or:

"capital gain = Form 11".

The correct outcome depends on the user's complete facts and the rules for the relevant tax year.

==================================================
5. INVESTMENT CLASSIFICATION
==================================================

Never assume that all investments are taxed under the same regime.

Before calculating the tax on an investment event, identify the instrument and determine its applicable Irish tax treatment.

In particular:

SHARES:
Determine whether the instrument is an ordinary share investment and apply the applicable CGT rules where appropriate.

ETFs / INVESTMENT FUNDS:
Do not automatically classify an ETF as an ordinary CGT asset.

Determine the relevant fund/investment classification, including domicile and any other facts required by the applicable Irish tax rules.

Consult authoritative Revenue guidance before selecting the calculation method.

OPTIONS / DERIVATIVES:
Determine what type of option or derivative is involved.

Distinguish, in particular, between:

- employee/share-option arrangements;
- exchange-traded investment options;
- other derivatives or financial instruments.

Do not assume that "option" means an employee share option or that it means an ordinary CGT asset.

If classification cannot be established from available evidence, do not calculate the tax result. Request the minimum information necessary to resolve the classification.

==================================================
6. EVIDENCE
==================================================

Prefer primary or authoritative evidence, including where available:

- Revenue records
- employment/PAYE records
- broker transaction records
- broker statements
- dividend statements
- fund documentation
- bank statements
- pension statements
- official tax documents

Do not blindly rely on a broker's tax classification or gain/loss calculation.

A broker's reported gain or loss is evidence, not necessarily the Irish taxable amount.

For every material tax calculation, maintain an audit trail containing:

- source/evidence identifier
- relevant transaction/event
- dates
- quantities
- amounts
- currency
- FX information where relevant
- asset classification
- tax regime
- applicable tax rule/source
- calculation inputs
- calculation result
- assumptions
- unresolved issues

==================================================
7. CAPITAL GAINS
==================================================

For transactions classified as CGT events:

- identify all relevant disposals;
- establish acquisition history;
- identify the applicable acquisition/matching rules;
- identify allowable costs;
- identify applicable reliefs/exemptions;
- identify current-year gains and losses;
- consider relevant prior-year losses;
- calculate the resulting taxable position;
- map the result to the appropriate reporting mechanism.

Use `run_cgt_computation` only after establishing that the transaction belongs in the CGT calculation workflow.

Do not use the CGT calculator merely because the user describes something as an "investment".

If transaction history is incomplete, do not invent acquisition dates, costs, FX rates, quantities or matching.

==================================================
8. OTHER INVESTMENT TAX REGIMES
==================================================

If an investment is not subject to the ordinary CGT calculation workflow:

1. identify the applicable Irish tax regime using `query_tax_knowledge`;
2. gather the necessary inputs;
3. use the appropriate calculation tool if available;
4. map the resulting tax information to the correct return section;
5. preserve the supporting evidence and calculation trail.

==================================================
9. TAX KNOWLEDGE
==================================================

Use `query_tax_knowledge` whenever the answer depends materially on:

- tax law
- tax-year-specific rules
- tax rates
- thresholds
- exemptions
- reliefs
- filing requirements
- payment deadlines
- asset classification
- ETF/fund treatment
- offshore fund treatment
- share matching rules
- options or derivatives
- employee share schemes
- Revenue form requirements

Prefer authoritative Revenue sources and current tax-year-specific guidance.

Do not invent a rule because it appears plausible.

If the available sources do not resolve an issue, mark it as REQUIRES_REVIEW rather than making an unsupported assumption.

==================================================
10. CALCULATIONS
==================================================

Use `calculate` for general arithmetic.

Use specialized calculation tools where appropriate.

Do not perform material tax calculations mentally.

Preserve the inputs, formula, result, currency/units and relevant rounding in the audit trail.

==================================================
11. FORM STATE
==================================================

Always begin by reading the current `form_state`.

Determine:

- target tax year;
- currently selected form/reporting mechanism;
- fields already populated;
- fields already verified;
- unresolved sections;
- information already gathered.

Do not overwrite verified information without identifying the conflict.

If new evidence conflicts with existing form data, flag the discrepancy and resolve it before overwriting.

==================================================
12. UPDATING FORM FIELDS
==================================================

Only populate a form field after the underlying tax conclusion has been established.

Use `update_form_field`.

The `rationale` must be concise but auditable and should identify:

- evidence/source used;
- relevant classification;
- applicable tax rule/source;
- calculation or derivation;
- important assumptions.

Do not put unnecessary sensitive raw financial data into the rationale if a source/evidence identifier is sufficient.

==================================================
13. MISSING INFORMATION
==================================================

Never invent missing facts.

If a required fact is missing, ask the smallest number of targeted questions necessary to continue.

Examples include:

- ETF/fund identifier or domicile;
- acquisition date;
- acquisition cost;
- disposal proceeds;
- transaction costs;
- option type;
- whether an option is employment-related;
- exercise/assignment/release date;
- whether payroll tax was already withheld;
- prior-year capital losses;
- foreign-currency transaction information.

Do not ask for information that is irrelevant to the applicable tax treatment.

==================================================
14. COMPLETENESS REVIEW
==================================================

Before declaring the preparation complete, perform an independent completeness check.

Verify:

- correct tax year;
- correct filing mechanism;
- PAYE information considered;
- all applicable income categories considered;
- investment activity considered;
- all relevant disposals considered;
- shares classified correctly;
- ETFs/funds classified before calculation;
- options/derivatives classified before calculation;
- foreign income considered where relevant;
- current and relevant prior-year losses considered;
- relevant reliefs/deductions considered;
- CGT reporting completed where applicable;
- separate CGT payment obligations considered;
- material calculations supported by evidence;
- no invented values;
- no unresolved required fields hidden as completed.

The task is complete only when every applicable obligation is:

- VERIFIED and populated;
- explicitly NOT_APPLICABLE;
- or explicitly REQUIRES_INFORMATION / REQUIRES_REVIEW.

==================================================
15. ESCALATION
==================================================

Escalate for review when:

- tax classification is ambiguous;
- authoritative sources do not resolve the issue;
- transaction records are incomplete;
- trading/business activity may need to be distinguished from investment activity;
- an unusual financial instrument is involved;
- an employee share scheme has unusual terms;
- multiple tax regimes may apply;
- evidence sources conflict materially;
- a previous return may be incorrect;
- the tax result depends on facts that have not been established.

Never conceal uncertainty.

The goal is not to fill every form field.

The goal is to produce a complete, evidence-backed, tax-year-correct and auditable tax filing, while clearly identifying anything that requires additional information or review.

==================================================
16. TOOL CONVENTIONS & SCHEMAS
==================================================

When calling tools, always adhere strictly to the expected literal identifiers:
- `set_filing_form`:
  - `form_type`: Must be strictly one of `"form11"`, `"form12"`, or `"cg1"`. Do not pass full titles or extra text.
- `update_form_field`:
  - `section`: Must be strictly one of `"capital_gains"`, `"income"`, `"tax_credits"`, or `"additional_fields"`.
  - `method`: Must be strictly one of `"computed_via_tool"`, `"computed_via_rag"`, or `"user_override"`.
- `clear_form_field`:
  - `section`: Must be strictly one of `"capital_gains"`, `"income"`, `"tax_credits"`, or `"additional_fields"`.
"""


def _apply_form_field_update(  # noqa: PLR0917
    form: IrishTaxFormState,
    section: TaxFormSection,
    field_name: str,
    value: FieldValueType,
    rationale: str,
    method: FieldStatus,
) -> tuple[bool, str]:
    """Apply an update to a specific section and field of the Irish tax form state.

    Args:
        form: Active IrishTaxFormState instance.
        section: Form section key ('capital_gains', 'income', or 'additional_fields').
        field_name: Identifier for the form field.
        value: Computed or entered field value.
        rationale: Explanatory reasoning and arithmetic traceability.
        method: Method tag describing how the field was computed.

    Returns:
        Tuple of (success_boolean, status_message).
    """
    canonical_section = normalize_form_section(section)
    canonical_method = normalize_field_status(method)

    if isinstance(form, IrishUndeterminedState) and canonical_section != "additional_fields":
        return (
            False,
            "Error: Filing obligation is currently undetermined. "
            "Use 'set_filing_form' first to select 'form11', 'form12' or 'cg1' before adding section fields.",
        )
    if isinstance(form, IrishCG1State) and canonical_section in ("income", "tax_credits"):
        return (
            False,
            f"Error: Form CG1 is for Capital Gains Tax only and does not contain a '{canonical_section}' section. "
            "Use 'set_filing_form' to switch form if Income Tax reporting is required.",
        )

    if canonical_section == "capital_gains" and isinstance(form, (IrishForm11State, IrishForm12State, IrishCG1State)):
        target_dict = form.capital_gains
    elif canonical_section == "income" and isinstance(form, (IrishForm11State, IrishForm12State)):
        target_dict = form.income
    elif canonical_section == "tax_credits" and isinstance(form, IrishForm12State):
        target_dict = form.tax_credits
    elif canonical_section == "additional_fields":
        target_dict = form.additional_fields
    else:
        return False, f"Error: Invalid section '{section}' for form {form.form_type}."

    if field_name in target_dict and target_dict[field_name].status == "user_override":
        return True, f"Field '{field_name}' is locked (user_override). Skipped."

    target_dict[field_name] = FormField(
        name=field_name,
        value=value,
        status=canonical_method,
        rationale=rationale,
    )
    return True, f"Successfully updated {canonical_section}.{field_name} = {value}"


def _apply_form_field_clear(
    form: IrishTaxFormState,
    section: TaxFormSection,
    field_name: str,
) -> tuple[bool, str]:
    """Clear or remove a field from the Irish tax form state.

    Args:
        form: Active IrishTaxFormState instance.
        section: Form section key ('capital_gains', 'income', 'tax_credits', or 'additional_fields').
        field_name: Identifier for the form field to remove.

    Returns:
        Tuple of (success_boolean, status_message).
    """
    canonical_section = normalize_form_section(section)

    if isinstance(form, IrishUndeterminedState) and canonical_section != "additional_fields":
        return False, f"Section '{section}' is not present on undetermined form state."
    if isinstance(form, IrishCG1State) and canonical_section in ("income", "tax_credits"):
        return False, f"Form CG1 does not contain a '{section}' section."

    if canonical_section == "capital_gains" and isinstance(form, (IrishForm11State, IrishForm12State, IrishCG1State)):
        target_dict = form.capital_gains
    elif canonical_section == "income" and isinstance(form, (IrishForm11State, IrishForm12State)):
        target_dict = form.income
    elif canonical_section == "tax_credits" and isinstance(form, IrishForm12State):
        target_dict = form.tax_credits
    elif canonical_section == "additional_fields":
        target_dict = form.additional_fields
    else:
        return False, f"Invalid section '{section}'."

    if field_name not in target_dict:
        return True, f"Field '{field_name}' not present in {section} (nothing to clear)."

    if target_dict[field_name].status == "user_override":
        return False, f"Field '{field_name}' is locked (user_override) and cannot be cleared by agent."

    del target_dict[field_name]
    return True, f"Successfully cleared {canonical_section}.{field_name}"


def _apply_form_type_switch(
    current_form: IrishTaxFormState,
    form_type: FilingFormType,
    rationale: str,
    is_chargeable_person: bool | None,
    has_cgt_obligation: bool | None,
) -> IrishTaxFormState:
    """Transition or update the tax form state based on the determined filing obligation.

    Args:
        current_form: Existing tax form state.
        form_type: Target filing form ('form11', 'form12', or 'cg1').
        rationale: Legal and factual rationale justifying form determination.
        is_chargeable_person: Optional boolean indicating self-assessment status.
        has_cgt_obligation: Optional boolean indicating CGT reporting status.

    Returns:
        New or updated IrishTaxFormState instance.
    """
    canonical_form_type = normalize_form_type(form_type)
    decision = FilingObligationDecision(
        required_form=canonical_form_type,
        is_chargeable_person=is_chargeable_person,
        has_cgt_obligation=has_cgt_obligation,
        rationale=rationale,
    )
    capital_gains = (
        current_form.capital_gains
        if isinstance(current_form, (IrishForm11State, IrishForm12State, IrishCG1State))
        else {}
    )
    income = current_form.income if isinstance(current_form, (IrishForm11State, IrishForm12State)) else {}
    tax_credits = current_form.tax_credits if isinstance(current_form, IrishForm12State) else {}
    additional_fields = current_form.additional_fields

    if canonical_form_type == "cg1":
        return IrishCG1State(
            tax_year=current_form.tax_year,
            obligation_decision=decision,
            capital_gains=capital_gains,
            additional_fields=additional_fields,
        )
    if canonical_form_type == "form12":
        return IrishForm12State(
            tax_year=current_form.tax_year,
            obligation_decision=decision,
            income=income,
            tax_credits=tax_credits,
            capital_gains=capital_gains,
            additional_fields=additional_fields,
        )
    return IrishForm11State(
        tax_year=current_form.tax_year,
        obligation_decision=decision,
        capital_gains=capital_gains,
        income=income,
        additional_fields=additional_fields,
    )


# ---------------------------------------------------------------------------
# Specialized Irish Tax Filing Agent Tools
# ---------------------------------------------------------------------------


@trace_tool(
    tool_name="get_form_state",
    summary_fn=lambda res: "Retrieved current tax form state.",
)
async def get_form_state(ctx: RunContext[TaxFilingDeps]) -> str:
    """Returns the current IrishTaxFormState (Undetermined, Form 11, Form 12, or Form CG1) as a JSON string."""
    return ctx.deps.form_state.model_dump_json(indent=2)


@trace_tool(
    tool_name="set_filing_form",
    summary_fn=lambda res: res,
)
async def set_filing_form(
    ctx: RunContext[TaxFilingDeps],
    form_type: FilingFormType,
    rationale: str,
    is_chargeable_person: bool | None = None,
    has_cgt_obligation: bool | None = None,
) -> str:
    """Set or update the applicable Irish tax filing form (Form 11 vs Form 12 vs Form CG1).

    Args:
        ctx: RunContext containing tax filing dependencies.
        form_type: Exact literal string: 'form11', 'form12', or 'cg1'.
        rationale: Explanation of why this form is required, citing Revenue rules and taxpayer facts.
        is_chargeable_person: Optional boolean indicating if taxpayer is a chargeable person.
        has_cgt_obligation: Optional boolean indicating if taxpayer has CGT reporting obligations.
    """
    canonical_form_type = normalize_form_type(form_type)
    effective_form_type: Literal["form11", "form12", "cg1"] = (
        "form11" if canonical_form_type == "undetermined" else canonical_form_type
    )
    ctx.deps.form_state = _apply_form_type_switch(
        current_form=ctx.deps.form_state,
        form_type=effective_form_type,
        rationale=rationale,
        is_chargeable_person=is_chargeable_person,
        has_cgt_obligation=has_cgt_obligation,
    )
    return f"Successfully set tax filing form to {effective_form_type.upper()}."


@trace_tool(
    tool_name="update_form_field",
    summary_fn=lambda res: res,
)
async def update_form_field(
    ctx: RunContext[TaxFilingDeps],
    section: TaxFormSection,
    field_name: str,
    value: FieldValueType,
    rationale: str,
    method: FieldStatus,
) -> str:
    """Update a field on the tax form.

    Args:
        ctx: RunContext containing tax filing dependencies.
        section: Exact literal string: 'capital_gains', 'income', 'tax_credits', or 'additional_fields'.
        field_name: Internal identifier for the field (e.g. 'PAYE_Income', 'Chargeable_Gain').
        value: Computed or entered value (e.g. string, number, or boolean).
        rationale: Explanation for this value. If computed via RAG or calculate, include expressions and data.
        method: Exact literal string: 'computed_via_tool', 'computed_via_rag', or 'user_override'.
    """
    _, message = _apply_form_field_update(
        form=ctx.deps.form_state,
        section=section,
        field_name=field_name,
        value=value,
        rationale=rationale,
        method=method,
    )
    return message


@trace_tool(
    tool_name="clear_form_field",
    summary_fn=lambda res: res,
)
async def clear_form_field(
    ctx: RunContext[TaxFilingDeps],
    section: TaxFormSection,
    field_name: str,
) -> str:
    """Clear or remove a previously populated field from the tax form if computation is invalidated.

    Args:
        ctx: RunContext containing tax filing dependencies.
        section: Exact literal string: 'capital_gains', 'income', 'tax_credits', or 'additional_fields'.
        field_name: Internal identifier for the field to remove.
    """
    _, message = _apply_form_field_clear(
        form=ctx.deps.form_state,
        section=section,
        field_name=field_name,
    )
    return message


@trace_tool(
    tool_name="run_cgt_computation",
    summary_fn=lambda res: "Executed deterministic CGT computation engine.",
)
async def run_cgt_computation(ctx: RunContext[TaxFilingDeps], tax_year: int) -> str:
    """Runs the deterministic Capital Gains Tax engine for the given tax year.

    Returns a summary of proceeds, net gains, and quarantined losses.
    You can use this to populate the 'capital_gains' section of the form.
    """
    db = ctx.deps.db
    try:
        records = db.filter_financial_records(
            purchase_date_start=datetime(tax_year, 1, 1),
            purchase_date_end=datetime(tax_year, 12, 31, 23, 59, 59),
            logic="AND",
        )
        sells = [r for r in records if isinstance(r, TradeRecord) and r.action == TransactionAction.SELL]

        total_proceeds = 0.0
        total_net_gain = 0.0
        total_quarantined = 0.0

        for sell in sells:
            if sell.id is None:
                continue

            proceeds, net_gain, quarantined = _process_cgt_disposal(db, sell.id)

            total_proceeds += proceeds
            total_net_gain += net_gain
            total_quarantined += quarantined

        summary = {
            "total_disposal_proceeds": total_proceeds,
            "total_unrestricted_net_gain": total_net_gain,
            "total_quarantined_loss": total_quarantined,
        }
        return json.dumps(summary, indent=2)

    except Exception as e:
        return f"Error running CGT computation: {str(e)}"


@trace_tool(
    tool_name="run_pension_calculator",
    summary_fn=lambda res: "Computed pension relief limits.",
)
async def run_pension_calculator(ctx: RunContext[TaxFilingDeps], age: int, net_relevant_earnings: float) -> str:
    """Calculate the max tax-relievable pension contribution for Ireland based on age and earnings."""
    try:
        res = calculate_pension_relief_limit(age, net_relevant_earnings)
        return res.model_dump_json(indent=2)
    except Exception as e:
        return f"Error computing pension relief: {str(e)}"


def _register_tax_filing_tools(agent: Agent[TaxFilingDeps, str]) -> None:
    """Register reusable tax tools and specialized Irish tax filing tools on agent.

    Args:
        agent: Target PydanticAI agent instance.
    """
    register_tax_tools(agent)
    agent.tool(retries=2)(get_form_state)
    agent.tool(retries=2)(set_filing_form)
    agent.tool(retries=2)(update_form_field)
    agent.tool(retries=2)(clear_form_field)
    agent.tool(retries=2)(run_cgt_computation)
    agent.tool(retries=2)(run_pension_calculator)


def create_tax_filing_agent() -> Agent[TaxFilingDeps, str]:
    """Create the specialized Irish tax filing agent with all tools registered."""
    model = build_pydantic_model("TAX_FILING")

    agent = Agent(
        model=model,
        deps_type=TaxFilingDeps,
        output_type=str,
        system_prompt=TAX_FILING_SYSTEM_PROMPT,
        retries=2,
    )
    _register_tax_filing_tools(agent)
    return agent
