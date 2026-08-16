"""Courtroom deliberation session orchestrator.

Manages the four-round tax compliance debate between Plaintiff, Defense, and Judge
agents, all backed by PydanticAI. Evidence is loaded once at construction via
semantic search and injected into every agent turn through ``CourtDeps``.
"""

import logging

from pydantic_ai import Agent
from pydantic_ai.models import Model

from backend.db_manager import DatabaseManager
from backend.deliberation.court_tools import register_tools
from backend.deliberation.models import CourtVerdict, DebateResult, EvidenceChunk
from backend.deliberation.pydantic_agents import (
    CourtDeps,
    make_defense_agent,
    make_judge_agent,
    make_plaintiff_agent,
)
from backend.llm.embedding_runner import BaseEmbeddingRunner
from backend.llm.pydantic_utils import (
    get_cache_settings,
    log_agent_usage,
)

logger = logging.getLogger(__name__)


class CourtroomSession:
    """Orchestrates a multi-agent tax compliance courtroom deliberation.

    The session loads regulatory evidence from the vector database at construction
    time, then runs a structured four-round debate between three PydanticAI agents:

    1. **Plaintiff (Tax Filing Strategist):** Proposes an optimized filing strategy
       grounded in the retrieved regulatory evidence.
    2. **Defense (Tax Auditor):** Audits the proposal and raises compliance objections.
    3. **Plaintiff Rebuttal:** Responds to the auditor's objections with legal arguments.
    4. **Judge (Compliance Arbiter):** Issues the final ruling and returns a validated
       ``VerificationBlock`` via PydanticAI structured output — no manual JSON parsing.

    Each agent has access to three tools via ``court_tools.register_tools``:
    - ``search_evidence``: Semantic KNN search over the regulatory DB.
    - ``get_chunk``: Retrieve a chunk by primary key.
    - ``calculate``: Sandboxed arithmetic evaluator for verified tax math.

    Evidence retrieved during ``_load_evidence`` forms a stable prompt prefix injected
    into every agent turn, enabling prompt-cache reuse in caching-aware providers.

    Args:
        scenario_name: Short display name for the tax scenario (used in logs and prompts).
        scenario_description: Full narrative description of the taxpayer's situation.
        jurisdiction: Jurisdiction identifier (e.g. ``'italy'``, ``'ireland'``).
        db: Initialized DatabaseManager connected to the SQLite database.
        embedding_runner: BGE-M3 embedding runner for semantic evidence retrieval.
        plaintiff_model: PydanticAI Model for the Plaintiff agent.
        defense_model: PydanticAI Model for the Defense agent.
        judge_model: PydanticAI Model for the Judge agent.
        search_keywords: Key-phrases describing the regulatory topics. Each phrase
            is embedded and searched independently to maximize evidence coverage.

            .. todo::
                Replace with LLM-based keyword extraction from ``scenario_description``
                so keywords do not need to be manually specified by the caller.

        evidence_limit: Maximum number of evidence chunks per keyword search (default: 10).

    Raises:
        ValueError: If no evidence is found for the given jurisdiction and keywords.
    """

    def __init__(
        self,
        scenario_name: str,
        scenario_description: str,
        jurisdiction: str,
        db: DatabaseManager,
        embedding_runner: BaseEmbeddingRunner,
        plaintiff_model: Model,
        defense_model: Model,
        judge_model: Model,
        search_keywords: list[str],
        evidence_limit: int = 10,
    ):
        self.scenario_name = scenario_name
        self.scenario_description = scenario_description
        self.jurisdiction = jurisdiction
        self.db = db
        self.embedding_runner = embedding_runner

        # Build PydanticAI agents
        self.plaintiff_agent: Agent[CourtDeps, str] = make_plaintiff_agent(plaintiff_model)
        self.defense_agent: Agent[CourtDeps, str] = make_defense_agent(defense_model)
        self.judge_agent: Agent[CourtDeps, CourtVerdict] = make_judge_agent(judge_model)

        # Register tools on all agents
        register_tools(self.plaintiff_agent, self.defense_agent, self.judge_agent)

        # Evidence pool populated once at construction via semantic search
        self.retrieved_evidence: list[EvidenceChunk] = []
        self._load_evidence(search_keywords, evidence_limit)

    def _load_evidence(self, keywords: list[str], limit: int) -> None:
        """Embed each key-phrase individually and retrieve evidence chunks via KNN search.

        Runs one semantic search per key-phrase to ensure each topic is represented
        in the evidence pool. Results are merged, deduplicated by chunk ID, sorted by
        ascending cosine distance, and trimmed to ``limit``.

        Args:
            keywords: List of regulatory key-phrases to search independently.
            limit: Maximum number of evidence chunks to retain after deduplication.

        Raises:
            ValueError: If no chunks are found for the jurisdiction and keywords.
        """
        logger.info(
            "🏛️  Executing semantic evidence retrieval for %d key-phrases: %s",
            len(keywords),
            keywords,
        )

        seen_ids: set[int] = set()
        merged: list[EvidenceChunk] = []

        for phrase in keywords:
            phrase_embedding = self.embedding_runner.embed(phrase)
            results = self.db.semantic_search(
                query_embedding=phrase_embedding,
                limit=limit,
                jurisdiction=self.jurisdiction,
            )
            for chunk in results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    merged.append(chunk)

        if not merged:
            raise ValueError(
                f"No evidence found for jurisdiction '{self.jurisdiction}' "
                + f"with keywords {keywords}. Cannot run debate without grounding facts."
            )

        # Sort by distance ascending (best semantic match first) and cap to limit
        merged.sort(key=lambda c: c.distance if c.distance is not None else float("inf"))
        self.retrieved_evidence = merged[:limit]

        logger.info(
            "🏛️  Loaded %d regulatory manual sections into the courtroom evidence pool.",
            len(self.retrieved_evidence),
        )

    def _compile_debate_context_prefix(self) -> str:
        """Build the shared context prefix injected into every agent's prompt.

        Contains the scenario details and the retrieved evidence pool. Identical
        across all four debate rounds to enable prompt-cache reuse.

        Returns:
            A formatted string with scenario description and all evidence chunks.
        """
        parts: list[str] = []
        parts.append(f"# COURTROOM DELIBERATION CASE: {self.scenario_name.upper()}\n")
        parts.append(
            "## 1. TAXPAYER SCENARIO DETAILS (Factual Evidence):\n" + f"```text\n{self.scenario_description}\n```\n"
        )
        parts.append("## 2. RETRIEVED REGULATORY COMPLIANCE MANUALS (Evidence Pool):\n")

        for rank, ev in enumerate(self.retrieved_evidence):
            content = ev.parent_text_content if ev.parent_text_content else ev.text_content
            parts.append(
                f"### [Reference {rank + 1}] Document: {ev.document_name} "
                + f"| Page: {ev.page_number} | Source: {ev.source_type.upper()} "
                + f"| Confidence: {ev.confidence_level.upper()} "
                + f"| Chunk Index: {ev.chunk_index}\n"
                + f"```text\n{content}\n```\n"
            )

        parts.append("\n## 3. CHRONOLOGICAL COURTROOM DEBATE TRANSCRIPT:\n")
        return "\n".join(parts)

    def run_debate(self) -> DebateResult:
        """Orchestrate the four-round courtroom debate and return the full result.

        Rounds:
            1. Plaintiff files an optimized tax proposal.
            2. Defense audits the proposal and raises objections.
            3. Plaintiff rebuts the objections.
            4. Judge issues the final compliance ruling with a structured CourtVerdict.

        Each agent has tool access to ``search_evidence``, ``get_chunk``, and
        ``calculate`` for dynamic evidence lookup and verified arithmetic.

        Returns:
            A ``DebateResult`` with the full transcript, verdict text, and validated
            ``CourtVerdict`` (structured output enforced by PydanticAI).
        """
        logger.info("\n%s", "=" * 50)
        logger.info("🏛️  STARTING DEBATE: %s", self.scenario_name)
        logger.info("%s\n", "=" * 50)

        deps = CourtDeps(
            db=self.db,
            embedding_runner=self.embedding_runner,
            jurisdiction=self.jurisdiction,
        )

        prefix = self._compile_debate_context_prefix()
        debate_log: list[str] = []

        # --- ROUND 1: Plaintiff Proposal ---
        logger.info("📣 [Round 1] Plaintiff Counsel drafting optimized tax filing strategy...")
        plaintiff_result = self.plaintiff_agent.run_sync(
            prefix + "* Plaintiff Filing Proposal:\n",
            deps=deps,
            model_settings=get_cache_settings(self.plaintiff_agent.model),
        )
        plaintiff_proposal = plaintiff_result.output
        log_agent_usage("Round 1 - Plaintiff Proposal", plaintiff_result, logger=logger)
        logger.debug("\n🧑‍⚖️ [Plaintiff Proposal]:\n%s\n%s\n%s\n", "-" * 40, plaintiff_proposal, "-" * 40)
        debate_log.append(f"### Plaintiff Filing Proposal:\n{plaintiff_proposal}\n")

        # --- ROUND 2: Defense Objection ---
        logger.info("📣 [Round 2] Defense Counsel auditing the filing strategy...")
        defense_result = self.defense_agent.run_sync(
            prefix + "\n".join(debate_log) + "\n* Defense Audit Objections:\n",
            deps=deps,
            model_settings=get_cache_settings(self.defense_agent.model),
        )
        defense_objection = defense_result.output
        log_agent_usage("Round 2 - Defense Objection", defense_result, logger=logger)
        logger.debug("\n🧑‍⚖️ [Defense Objections]:\n%s\n%s\n%s\n", "-" * 40, defense_objection, "-" * 40)
        debate_log.append(f"### Defense Audit Objections:\n{defense_objection}\n")

        # --- ROUND 3: Plaintiff Rebuttal ---
        logger.info("📣 [Round 3] Plaintiff Counsel drafting legal rebuttal...")
        rebuttal_result = self.plaintiff_agent.run_sync(
            prefix + "\n".join(debate_log) + "\n* Plaintiff Legal Rebuttal:\n",
            deps=deps,
            model_settings=get_cache_settings(self.plaintiff_agent.model),
        )
        plaintiff_rebuttal = rebuttal_result.output
        log_agent_usage("Round 3 - Plaintiff Rebuttal", rebuttal_result, logger=logger)
        logger.debug("\n🧑‍⚖️ [Plaintiff Rebuttal]:\n%s\n%s\n%s\n", "-" * 40, plaintiff_rebuttal, "-" * 40)
        debate_log.append(f"### Plaintiff Legal Rebuttal:\n{plaintiff_rebuttal}\n")

        # --- ROUND 4: Judge Ruling (structured output) ---
        logger.info("📣 [Round 4] Neutral Judge deliberating and compiling the final ruling...")
        judge_result = self.judge_agent.run_sync(
            prefix + "\n".join(debate_log) + "\n* Neutral Judge Compliance Ruling:\n",
            deps=deps,
            model_settings=get_cache_settings(self.judge_agent.model),
        )

        court_verdict: CourtVerdict = judge_result.output
        verdict_text = str(judge_result.output)

        log_agent_usage("Round 4 - Judge Ruling", judge_result, logger=logger)
        logger.debug("\n🧑‍⚖️ [Judicial Verdict]:\n%s\n%s\n%s\n", "-" * 40, verdict_text, "-" * 40)
        debate_log.append(f"### Neutral Judge Compliance Ruling:\n{verdict_text}\n")

        full_transcript = prefix + "\n".join(debate_log)

        return DebateResult(
            full_transcript=full_transcript,
            verdict=verdict_text,
            court_verdict=court_verdict,
        )
