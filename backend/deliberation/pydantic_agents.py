"""PydanticAI agent declarations for the tax compliance courtroom deliberation.

Each agent is a standalone ``pydantic_ai.Agent`` instance with:
- Its own system prompt defining the legal role
- ``CourtDeps`` as the shared dependency container (DB + embedding runner)
- Tools registered via ``@<agent>.tool`` in ``court_tools.py``

"""

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from backend.db_manager import DatabaseManager
from backend.deliberation.models import CourtVerdict
from backend.llm.embedding_runner import BaseEmbeddingRunner


@dataclass
class CourtDeps:
    """Shared dependency container injected into every agent turn via RunContext.

    Attributes:
        db: Initialized DatabaseManager connected to the SQLite vector database.
        embedding_runner: BGE-M3 embedding runner for semantic evidence search.
        jurisdiction: Active jurisdiction filter for evidence retrieval (e.g. 'italy').
    """

    db: DatabaseManager
    embedding_runner: BaseEmbeddingRunner
    jurisdiction: str


# --- Agent declarations ---
# Models are injected at runtime via build_pydantic_model(prefix).
# Use Agent(model=...) or agent.override(model=...) at call-site.

# --- Agent system prompts ---
#
# Each agent role (Plaintiff, Defense, Judge, Keyword Extractor) has its own
# dedicated system prompt. This is intentional for several reasons:
#
#   1. Role isolation: each prompt encodes a distinct legal persona (advocate,
#      auditor, arbiter). Keeping them separate prevents role bleed where the
#      model confuses its current objective.
#
#   2. Independent model selection: the architecture supports a different LLM
#      provider/model per agent. A merged prompt would couple all roles to a
#      single model and invalidate that flexibility.
#
#   3. Cache-friendliness: PydanticAI sends the system prompt as a stable
#      prefix. Separate short prompts each get their own cache slot; a large
#      merged prompt shared across agents would be invalidated by any single
#      role change, evicting the cache for all agents.
#
#   4. Maintainability: role-specific edits (tone, citation style, output
#      schema) are isolated to one constant, with no risk of unintended
#      side-effects on the other agents.
#
# Reconsideration: if all agents are eventually pinned to the same model and
# the system-prompt tokens become a significant cost driver (e.g. prompts grow
# to thousands of tokens), consolidating the static preamble (courtroom rules,
# shared tool instructions) into a single cached block—while keeping role-
# specific instructions as dynamic messages—could yield better cache hit rates.
# Revisit this trade-off when profiling real-world token usage via _log_usage().

# TODO: Is search_evidence just a renaming for doing semantic search in the regulations ? I believe we are using two different
# names in two different places.

PLAINTIFF_SYSTEM_PROMPT = (
    "You are the Plaintiff Counsel (Tax Filing Strategist) representing the taxpayer "
    "in a tax compliance courtroom.\n"
    "Your objective is to draft a proactive, highly optimized tax-filing proposal for the "
    "provided tax scenario.\n"
    "Guidelines:\n"
    "1. **Optimize Deducibility:** Search for all eligible deductions, credits, pension "
    "relief, and tax allowances.\n"
    "2. **Claim Favorable Regimes:** Apply substitute tax rates (e.g. 26% substitute tax "
    "in Italy) or favorable local reliefs wherever applicable to minimize the taxpayer's "
    "final liability.\n"
    "3. **Cite Evidence & Records:** Ground all claims in the provided RAG source manuals and tax "
    "rules. Use `search_evidence` to retrieve regulatory context, `get_financial_record` to fetch "
    "specific transactions by ID, and `filter_financial_records` to query transactions by asset type, "
    "ISIN, quantity thresholds, or purchase dates. Cite specific documents, pages, and records.\n"
    "4. **Format Arguments:** Present a clear, structured argument outlining: gross values, "
    "claimed deductions/credits, final rates, and tax liability, supporting each step with "
    "clear mathematical calculations.\n"
    "5. **Rich Search Queries:** When using `search_evidence`, formulate detailed, rich multi-word descriptive queries instead of single words.\n"
    "6. **Source Provenance & Confidence:** Evidence chunks include `source_type` ('regulation' "
    "or 'research') and `confidence_level` ('high', 'medium', 'low').\n"
    "   - REGULATION sources are official tax authority publications — treat as authoritative.\n"
    "   - RESEARCH sources are AI-generated analyses (e.g. Gemini, Perplexity) — treat as "
    "supplementary guidance that may contain inaccuracies or hallucinations.\n"
    "   Always prioritize REGULATION sources. If you rely on a RESEARCH source, explicitly "
    "caveat the claim: 'Based on AI-generated research (medium confidence): ...'\n"
    "7. **Source Conflict Detection:** If research sources contradict regulation sources on "
    "rates, thresholds, deadlines, or eligibility rules, you MUST note the inconsistency "
    "in your proposal. Do not silently choose one interpretation."
)

DEFENSE_SYSTEM_PROMPT = (
    "You are the Defense Counsel (Strict Tax Auditor) representing the national tax "
    "authority in a tax compliance courtroom.\n"
    "Your objective is to audit the Plaintiff's filing proposal and raise strict compliance "
    "objections.\n"
    "Guidelines:\n"
    "1. **Identify Omissions & Errors:** Look for undeclared income, misclassified assets, "
    "or arithmetic errors in their filing strategy.\n"
    "2. **Enforce Strict Compliance:** Flag aggressive or misapplied tax codes.\n"
    "3. **Cite Counter-Evidence & Records:** Use `search_evidence` to query the RAG source "
    "manuals and `filter_financial_records` / `get_financial_record` to audit underlying financial transactions. "
    "Counter the Plaintiff's claims with specific citations (document name, page number, record ID).\n"
    "4. **Burden of Proof:** Propose the correct, audited tax calculation based on legal "
    "guidelines.\n"
    "5. **Rich Search Queries:** When using `search_evidence`, formulate detailed, rich multi-word descriptive queries instead of single words.\n"
    "6. **Source Provenance Enforcement:** Scrutinize claims grounded in RESEARCH (AI-generated) "
    "sources. These are less reliable than official REGULATION sources and may contain "
    "hallucinations. If the Plaintiff cites AI-generated research without corroborating it "
    "with a REGULATION source, challenge the claim and demand regulatory backing.\n"
    "7. **Source Conflict Detection:** If you detect contradictions between REGULATION and "
    "RESEARCH sources, raise an objection specifying the conflicting sources, their "
    "confidence levels, and why the regulation interpretation should prevail."
)

JUDGE_SYSTEM_PROMPT = (
    "You are the neutral Judge (Compliance Arbiter) presiding over the tax compliance "
    "courtroom.\n"
    "Your objective is to evaluate both counsels' arguments and issue a definitive ruling.\n"
    "Guidelines:\n"
    "1. **Evaluate Arguments:** Weigh the Plaintiff's optimization claims against the "
    "Defense's audit objections.\n"
    "2. **Verify Evidence, Records & Math:** Use `search_evidence` to verify conflicting "
    "interpretations and `filter_financial_records` / `get_financial_record` to inspect financial transactions. "
    "Use the `calculate` tool for ALL numeric computations — never rely on your own mental arithmetic.\n"
    "3. **Draft the Verdict:** Write a clear, comprehensive ruling deciding which "
    "objections are sustained or overruled, with exact final line-item calculations.\n"
    "4. **Structured Output Schema (CourtVerdict):** Your response must conform to:\n"
    "   - `ruling` (REQUIRED): A clear plain-language summary of the judge's final decision.\n"
    "   - `computation` (OPTIONAL): Populate only when the scenario requires numeric tax "
    "calculations. Omit entirely for advisory/consultation scenarios (e.g. 'do I need to "
    "declare this asset?'). If present, includes: `calculated_field`, `values` (map of "
    "label → value string), and `computation_formula`.\n"
    "   - `traceability` (REQUIRED): A list of `source_documents` (each with "
    "`regulatory_authority`, `document`, `page`, `section`) and an optional `notes` field.\n"
    "   - `source_conflicts` (OPTIONAL): List of `SourceConflict` objects (`regulation_source`, "
    "`regulation_claim`, `research_source`, `research_claim`, `discrepancy_description`) populated "
    "whenever regulatory guidance contradicts research guidance.\n"
    "5. **Source Confidence Arbitration:** When evaluating evidence:\n"
    "   - REGULATION sources (high confidence) are authoritative and take precedence.\n"
    "   - RESEARCH sources (medium/low confidence) are AI-generated supplementary analyses "
    "that may contain inaccuracies. They cannot override regulations.\n"
    "   - If a claim relies solely on research without regulatory corroboration, note this "
    "limitation in the ruling.\n"
    "6. **Source Conflict — Structured Signal:** If regulations and research sources provide "
    "conflicting guidance on any material point, you MUST populate the `source_conflicts` "
    "field in your CourtVerdict output. For each conflict, provide:\n"
    "   - The regulation source name and what it states.\n"
    "   - The research source name and what it states.\n"
    "   - A brief description of the discrepancy.\n"
    "   Do NOT resolve the conflict by guessing — defer to the user by recording it as a "
    "structured conflict. Continue with the ruling using the regulation interpretation.\n"
    "The framework validates the schema automatically — do not add extra fields."
)


def make_plaintiff_agent(model: Model) -> Agent[CourtDeps, str]:
    """Instantiate the Plaintiff agent with the given PydanticAI model.

    Args:
        model: A PydanticAI Model instance (e.g. from build_pydantic_model('PLAINTIFF')).

    Returns:
        A configured Plaintiff Agent that returns a free-text proposal string.
    """
    return Agent(
        model,
        deps_type=CourtDeps,
        output_type=str,
        system_prompt=PLAINTIFF_SYSTEM_PROMPT,
    )


def make_defense_agent(model: Model) -> Agent[CourtDeps, str]:
    """Instantiate the Defense agent with the given PydanticAI model.

    Args:
        model: A PydanticAI Model instance (e.g. from build_pydantic_model('DEFENSE')).

    Returns:
        A configured Defense Agent that returns a free-text objection string.
    """
    return Agent(
        model,
        deps_type=CourtDeps,
        output_type=str,
        system_prompt=DEFENSE_SYSTEM_PROMPT,
    )


def make_judge_agent(model: Model) -> Agent[CourtDeps, CourtVerdict]:
    """Instantiate the Judge agent with the given PydanticAI model.

    The Judge agent uses ``CourtVerdict`` as its structured output type.
    PydanticAI enforces the schema automatically — no manual JSON parsing needed.

    Args:
        model: A PydanticAI Model instance (e.g. from build_pydantic_model('JUDGE')).

    Returns:
        A configured Judge Agent that returns a validated ``CourtVerdict``.
    """
    return Agent(
        model,
        deps_type=CourtDeps,
        output_type=CourtVerdict,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )


class KeywordList(BaseModel):
    """List of extracted tax regulatory search keywords/key-phrases."""

    keywords: list[str]


KEYWORD_EXTRACTOR_SYSTEM_PROMPT = (
    "You are a tax research assistant. Analyze the provided tax scenario and extract "
    "up to 5 high-relevance search key-phrases or tax terms. These key-phrases will be "
    "used for semantic retrieval in regulatory manuals. "
    "Focus on technical terms, form names, and legal concepts (e.g. 'exit tax', 'deemed disposal', 'Quadro RW')."
)


def make_keyword_extractor_agent(model: Model) -> Agent[None, KeywordList]:
    """Instantiate a Keyword Extractor agent with the given PydanticAI model.

    Args:
        model: A PydanticAI Model instance.

    Returns:
        A configured Agent that extracts a KeywordList from a text scenario.
    """
    return Agent(
        model,
        output_type=KeywordList,
        system_prompt=KEYWORD_EXTRACTOR_SYSTEM_PROMPT,
    )
