"""Entry point for the multi-agent courtroom deliberation engine.

Each agent (Plaintiff, Defense, Judge, Keyword Extractor) can be independently
configured to use a different LLM provider via env-var prefixes:

    PLAINTIFF_PROVIDER=openai-compatible  PLAINTIFF_MODEL=...
    DEFENSE_PROVIDER=openai-compatible    DEFENSE_MODEL=...
    JUDGE_PROVIDER=vertex                 JUDGE_MODEL=gemini-2.0-flash
    KEYWORD_EXTRACTOR_PROVIDER=...        KEYWORD_EXTRACTOR_MODEL=...

Falls back to DEFAULT_LLM_* for any prefix that is not explicitly set.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Literal

import dotenv

from backend.cli_common import CommonConfigArgs, add_common_config_args, parse_typed_args
from backend.db_manager import DatabaseManager, LocalDb
from backend.deliberation.courtroom import CourtroomSession
from backend.deliberation.models import TaxScenario
from backend.deliberation.pydantic_agents import make_keyword_extractor_agent
from backend.ingestion.helpers import log_env_vars
from backend.llm.embedding_runner import BgeM3EmbeddingRunner
from backend.llm.runner_factory import build_pydantic_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Load environment variables from .env
_ = dotenv.load_dotenv()


class DeliberationCliArgs(CommonConfigArgs):
    """Strongly-typed arguments for courtroom deliberation CLI."""

    scenario_desc: str | None = None
    scenario_name: str = "Tax Court Case"
    scenario_file: str | None = None
    jurisdiction: Literal["italy", "ireland"] | None = None
    verbose: bool = False


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and configure the ArgumentParser for courtroom deliberation."""
    parser = argparse.ArgumentParser(description="Multi-Agent Courtroom Deliberation Engine (PROClaim & Tool-MAD)")
    _ = parser.add_argument("--scenario-desc", type=str, help="Full narrative description of the taxpayer scenario.")
    _ = parser.add_argument(
        "--scenario-name", type=str, default="Tax Court Case", help="Display name for the tax scenario."
    )
    _ = parser.add_argument(
        "--scenario-file",
        type=str,
        help="Path to a JSON file containing the scenario details (keys: name, description, jurisdiction).",
    )
    _ = parser.add_argument(
        "--jurisdiction",
        type=str,
        choices=["italy", "ireland"],
        help="Select the tax case jurisdiction ('italy' or 'ireland'). Required if --scenario-file is not used.",
    )
    add_common_config_args(parser)
    _ = parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parse_typed_args(parser, DeliberationCliArgs)
    app_config = args.resolve_app_config()

    log_level = "DEBUG" if args.verbose else "INFO"
    logging.getLogger("backend").setLevel(log_level)
    log_env_vars(logger)

    # 1. Parse and validate scenario details using Pydantic
    if args.scenario_file:
        if not os.path.exists(args.scenario_file):
            logger.error("Scenario file not found: %s", args.scenario_file)
            sys.exit(1)
        try:
            with open(args.scenario_file, encoding="utf-8") as f:
                scenario_obj = TaxScenario.model_validate_json(f.read())
            scenario_desc = scenario_obj.description
            jurisdiction = scenario_obj.jurisdiction
            scenario_name = scenario_obj.name
        except Exception as e:
            logger.error("Failed to parse or validate scenario file: %s", e)
            sys.exit(1)
    else:
        if not args.scenario_desc:
            logger.error("Error: Either --scenario-desc or --scenario-file must be provided.")
            sys.exit(1)
        if not args.jurisdiction:
            logger.error("Error: Jurisdiction must be provided via --jurisdiction.")
            sys.exit(1)
        try:
            scenario_obj = TaxScenario(
                name=args.scenario_name,
                description=args.scenario_desc,
                jurisdiction=args.jurisdiction,
            )
            scenario_desc = scenario_obj.description
            jurisdiction = scenario_obj.jurisdiction
            scenario_name = scenario_obj.name
        except Exception as e:
            logger.error("Validation error for scenario parameters: %s", e)
            sys.exit(1)

    # 2. Initialize Database Manager
    if not os.path.exists(str(app_config.db_path)):
        logger.error("Database file does not exist at '%s'. Please complete ingestion first.", app_config.db_path)
        print("Connecting to database...")
    db = DatabaseManager(
        db_config=LocalDb(
            db_path=app_config.db_path,
            vector_db_path=app_config.vector_db_path,
        )
    )

    # 3. Build per-agent PydanticAI models (each agent can use a different provider)
    try:
        plaintiff_model = build_pydantic_model("PLAINTIFF")
        defense_model = build_pydantic_model("DEFENSE")
        judge_model = build_pydantic_model("JUDGE")
        extractor_model = build_pydantic_model("KEYWORD_EXTRACTOR")
    except ValueError as e:
        logger.error("Failed to build LLM models: %s", e)
        sys.exit(1)

    # 4. Extract search keywords/key-phrases using Keyword Extractor Agent
    logger.info("Extracting search keywords/key-phrases from the scenario description...")
    extractor_agent = make_keyword_extractor_agent(extractor_model)
    try:
        extraction_result = extractor_agent.run_sync(scenario_desc)
        search_keywords = extraction_result.output.keywords
        logger.info("Extracted key-phrases for semantic search: %s", search_keywords)
    except Exception as e:
        logger.error("Failed to extract keywords using LLM: %s", e)
        sys.exit(1)

    # 5. Initialize local BGE-M3 embedding runner for semantic evidence retrieval
    embedding_runner = BgeM3EmbeddingRunner()

    try:
        # 6. Create Courtroom Session (evidence retrieval runs at construction time)
        session = CourtroomSession(
            scenario_name=scenario_name,
            scenario_description=scenario_desc,
            jurisdiction=jurisdiction,
            db=db,
            embedding_runner=embedding_runner,
            plaintiff_model=plaintiff_model,
            defense_model=defense_model,
            judge_model=judge_model,
            search_keywords=search_keywords,
        )

        # 7. Run Courtroom Debate Rounds
        result = session.run_debate()

        # 8. Save transcript locally inside <data_dir>/processed/<timestamp>/file_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = app_config.processed_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = output_dir / f"transcript_{jurisdiction}.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            _ = f.write(result.full_transcript)

        logger.info("\n%s", "=" * 60)
        logger.info("🏛️  COURTROOM SESSION COMPLETED SUCCESSFULLY!")
        logger.info("  * Transcript saved to: %s", transcript_path)
        logger.info("%s\n", "=" * 60)

        # Print high-level verdict summary to stdout for machine consumption
        cv = result.court_verdict
        print("\n📝 COURT VERDICT:")
        print(f"  Ruling: {cv.ruling}")

        if cv.computation:
            comp = cv.computation
            print(f"\n  📊 Computation — {comp.calculated_field}")
            for k, v in comp.values.items():
                print(f"    - {k}: {v}")
            print(f"  Formula: {comp.computation_formula}")

        print("\n  📚 Evidence Citations:")
        for idx, src in enumerate(cv.traceability.source_documents):
            print(
                f"    [{idx + 1}] {src.regulatory_authority} "
                + f"| Doc: {src.document} "
                + f"| Page {src.page} ({src.section})"
            )
        if cv.traceability.notes:
            print(f"  Notes: {cv.traceability.notes}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
