"""Command-Line Interface for Tax Transaction Ingestion Subsystem.

Provides CLI functionality to run multi-voter LLM consensus pipeline, stage transactions,
list records, and delete records.
"""

import argparse
import logging
import os
import sys

import dotenv
from sqlmodel import Session

from backend.db_manager import DatabaseManager, LocalDb
from backend.db_models import FinancialRecord, IngestedSourceDocument, StagedFinancialRecord
from backend.ingestion.directa_csv_parser import parse_directa_csv
from backend.ingestion.helpers import (
    IngestionDocument,
    calculate_sha256,
    extract_account_country_and_provider,
    log_env_vars,
)
from backend.ingestion.parser import get_parser_registry
from backend.ingestion.pii.models import LLMRedactionConfig, PIIPipelineConfig
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.ingestion.pipeline import TransactionPipeline, VerificationStatus
from backend.llm.runner import BaseLLMRunner
from backend.llm.runner_factory import build_runner
from backend.llm.runners import MockRunner

logger = logging.getLogger(__name__)

# Load environment variables from .env
dotenv.load_dotenv()

# Processing location defaults to EU for data residency unless overridden
PROCESSING_LOCATION: str = (
    os.environ.get("DATALAB_PROCESSING_LOCATION_TRANSACTIONS") or os.environ.get("DATALAB_PROCESSING_LOCATION") or "eu"
)


def print_record_details(r: FinancialRecord | StagedFinancialRecord) -> None:
    """Print financial transaction record details in a clean table format.

    Args:
        r: FinancialRecord or StagedFinancialRecord instance to display.
    """
    event_str = r.event_timestamp.strftime("%Y-%m-%d %H:%M") if r.event_timestamp else "N/A"
    asset_str = f"{(r.asset_type or 'N/A').upper()} ({r.symbol or 'N/A'})"
    action_str = (r.action or "N/A").upper()
    status_str = (r.verification_status or "N/A").upper()
    doc_name = r.source_file_name or "N/A"
    provider_name = r.provider or "N/A"

    print("┌" + "─" * 78 + "┐")
    print(f"│ ID: {r.id or 0:<10} │ Provider: {provider_name:<15} │ Doc: {doc_name:<28} │")
    print("├" + "─" * 78 + "┤")
    print(f"│ Event Date: {event_str:<18} │ Asset: {asset_str:<22} │ Action: {action_str:<13} │")
    print(f"│ Quantity: {str(r.quantity):<18} │ Price: {str(r.unit_price):<18} │ Currency: {r.currency:<8} │")
    print(f"│ Fees: {str(r.fees):<22} │ Total: {str(r.total_amount):<18} │ EUR Total: {str(r.local_total_amount):<9} │")
    print(f"│ Verification Status: {status_str:<55} │")
    print("└" + "─" * 78 + "┘")


def _format_ocr_info(parser_name: str) -> str:
    if parser_name == "chandra":
        method = os.environ.get("CHANDRA_INFERENCE_METHOD", "").lower().strip()
        if not method:
            method = "vllm" if os.environ.get("VLLM_API_BASE") else "hf"
        if method == "vllm":
            base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
            model = os.environ.get("VLLM_MODEL_NAME", "datalab-to/chandra-ocr-2")
            return f"chandra (vLLM API: {base_url}, Model: {model})"
        else:
            device = os.environ.get("TORCH_DEVICE", "auto-detect")
            attn = os.environ.get("TORCH_ATTN", "sdpa")
            return f"chandra (Local PyTorch: {device}, Attn: {attn})"
    elif parser_name == "chandra_api":
        mode = os.environ.get("DATALAB_MODE", "balanced")
        return f"chandra_api (Datalab Cloud API, Mode: {mode})"
    else:
        return "pdfplumber (Standard layout & table extraction)"


def _format_pii_info(pii_pipeline: PIIPipeline) -> str:
    cfg = pii_pipeline.config
    presidio_str = "Enabled" if cfg.presidio_enabled else "Disabled"
    openai_str = "Enabled" if cfg.openai_filter_enabled else "Disabled"

    if cfg.llm_redaction:
        llm_str = cfg.llm_redaction.runner.model_name
    else:
        llm_str = "Disabled"

    return f"LLM Redaction: {llm_str} | Presidio: {presidio_str} | OpenAI Filter: {openai_str}"


def print_pipeline_summary(
    args_mode: str,
    parser_name: str,
    runners: list[BaseLLMRunner],
    pii_pipeline: PIIPipeline,
) -> None:
    """Print pipeline execution summary header."""
    ocr_info = _format_ocr_info(parser_name)
    pii_info = _format_pii_info(pii_pipeline)

    print("\n" + "=" * 80)
    print("📋 TRANSACTION INGESTION PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"📄 OCR Engine:           {ocr_info}")
    print(f"🛡️ PII Anonymization:    {pii_info}")
    print(f"🏛️ Voter Consensus:      Mode: {args_mode.upper()}")
    if len(runners) >= 3:
        print(f"   ├─ Voter 1:                  {runners[0].model_name}")
        print(f"   ├─ Voter 2:                  {runners[1].model_name}")
        print(f"   └─ Voter 3:                  {runners[2].model_name}")
    print("=" * 80 + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return argument parser with subcommands for transaction ingestion CLI."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    _ = parent_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database file. Defaults to TAX_DB_PATH env var.",
    )

    parser = argparse.ArgumentParser(
        description="Tax Transaction Ingestion Subsystem CLI",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available subcommands ('ingest', 'list', 'delete')",
    )

    # 1. Ingest subcommand
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest PDF/CSV transaction reports into database",
        parents=[parent_parser],
    )
    _ = p_ingest.add_argument(
        "--mode",
        type=str,
        choices=["api", "local", "mock"],
        default="api",
        help="Execution mode: 'api' / 'local', or 'mock' (offline test)",
    )
    _ = p_ingest.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Explicit opt-in flag required when using non-production test/mock modes",
    )

    _ = p_ingest.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest already loaded transaction documents",
    )
    _ = p_ingest.add_argument(
        "--force-pii-reprocessing",
        action="store_true",
        help="Force PII re-processing, ignoring cached anonymization results",
    )
    _ = p_ingest.add_argument(
        "--parser",
        type=str,
        choices=list(get_parser_registry().keys()),
        default="chandra",
        help="PDF extraction backend parser to use (default: 'chandra')",
    )
    _ = p_ingest.add_argument(
        "--force-ocr",
        "--force_ocr",
        dest="force_ocr",
        action="store_true",
        help="Force re-running OCR parser/API from scratch (bypasses local OCR cache)",
    )

    target_group = p_ingest.add_mutually_exclusive_group()
    _ = target_group.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to a single PDF/CSV transaction document to ingest",
    )
    _ = target_group.add_argument(
        "--folder",
        "--dir",
        type=str,
        default=None,
        help="Path to a folder containing PDF/CSV transaction documents to ingest",
    )
    _ = target_group.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Convenience alias: path to either a single file or a directory of documents to ingest",
    )

    # 2. List subcommand
    _ = subparsers.add_parser(
        "list",
        help="List all financial transactions in the database",
        parents=[parent_parser],
    )

    # 3. Delete subcommand
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete financial transaction records from the database",
        parents=[parent_parser],
    )
    del_group = p_delete.add_mutually_exclusive_group(required=True)
    _ = del_group.add_argument(
        "--id",
        "--delete-record",
        dest="delete_record",
        type=int,
        default=None,
        help="Delete specific record by ID",
    )
    _ = del_group.add_argument(
        "--all",
        "--delete-all",
        dest="delete_all",
        action="store_true",
        help="Delete all financial records from database",
    )

    return parser


def _handle_delete_record(db: DatabaseManager, record_id: int) -> None:
    """Handle deleting a single financial record by ID.

    Args:
        db: Active DatabaseManager instance.
        record_id: Integer ID of record to delete.
    """
    record = db.get_financial_record_by_id(record_id)
    if not record:
        print(f"❌ No record found with ID {record_id}")
        return
    print_record_details(record)
    confirm = input(f"Are you sure you want to delete transaction ID {record_id}? [y/N]: ").strip().lower()
    if confirm == "y":
        with Session(db.engine) as session:
            session.delete(record)
            session.commit()
        print(f"✅ Record {record_id} deleted successfully.")
    else:
        print("Aborted.")


def _handle_delete_all(db: DatabaseManager) -> None:
    """Handle deleting all financial records from database.

    Args:
        db: Active DatabaseManager instance.
    """
    records = db.get_financial_records()
    if not records:
        print("No records found to delete.")
        return
    confirm = input(f"Are you sure you want to delete ALL {len(records)} financial records? [y/N]: ").strip().lower()
    if confirm == "y":
        with Session(db.engine) as session:
            for r in records:
                session.delete(r)
            session.commit()
        print("✅ All financial records deleted successfully.")
    else:
        print("Aborted.")


def _handle_list(db: DatabaseManager) -> None:
    """Handle listing all financial records in database.

    Args:
        db: Active DatabaseManager instance.
    """
    records = db.get_financial_records()
    print(f"\nTotal Financial Records: {len(records)}")
    for r in records:
        print_record_details(r)


def _collect_target_files(target_path: str) -> list[tuple[str, str | None, str | None]]:
    """Collect (file_path, account_country, provider) tuples for ingestion.

    Args:
        target_path: File or directory path to ingest.

    Returns:
        List of (file_path, account_country, provider) tuples.
    """
    raw_files: list[str] = []
    if os.path.isfile(target_path):
        raw_files = [target_path]
    else:
        for root, _, files in os.walk(target_path):
            for file in files:
                raw_files.append(os.path.join(root, file))

    to_process: list[tuple[str, str | None, str | None]] = []
    for file_path in raw_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".pdf", ".csv"):
            continue

        try:
            detected_account_country, detected_provider = extract_account_country_and_provider(file_path)
        except ValueError:
            detected_account_country = None
            detected_provider = None

        to_process.append((file_path, detected_account_country, detected_provider))

    return to_process


def _handle_batch_ingestion(db: DatabaseManager, args: argparse.Namespace) -> None:
    """Handle batch document ingestion logic.

    Args:
        db: Active DatabaseManager instance.
        args: Parsed CLI arguments namespace.
    """
    runners: list[BaseLLMRunner] = []
    pii_pipeline: PIIPipeline = PIIPipeline()

    mode_str: str = str(args.mode)
    parser_str: str = str(args.parser)
    force_flag: bool = bool(args.force)
    force_pii_flag: bool = bool(args.force_pii_reprocessing)
    force_ocr_flag: bool = bool(args.force_ocr)

    if mode_str == "mock":
        print("🏛️  Initializing Consensus Pipeline in OFFLINE MOCK MODE...")
        runner = MockRunner()
        runners = [runner, runner, runner]
        pii_config = PIIPipelineConfig(llm_redaction=LLMRedactionConfig(runner=runner))
        pii_pipeline = PIIPipeline(config=pii_config)
    else:
        print("🏛️  Initializing Consensus Pipeline with Voter-specific API Clients...")
        for i in range(1, 4):
            try:
                runner = build_runner(f"VOTER_{i}")
                runners.append(runner)
            except ValueError as e:
                print(f"Error initializing Voter {i}: {e}", file=sys.stderr)
                sys.exit(1)

    pipeline = TransactionPipeline(db, runners, pii_pipeline=pii_pipeline)

    print_pipeline_summary(mode_str, parser_str, runners, pii_pipeline)

    target_path: str = str(args.file or args.folder or args.path or "data/raw_sources/records")

    if not os.path.exists(target_path):
        print(f"Error: Specified path '{target_path}' not found.", file=sys.stderr)
        return

    to_process = _collect_target_files(target_path=target_path)

    if not to_process:
        print("📁 No matching PDF or CSV files found to ingest.")
        return

    total_docs = len(to_process)
    print(f"\n📁 Found {total_docs} report file(s) to process.")

    for idx, (file_path, account_country, provider) in enumerate(to_process, start=1):
        is_csv = file_path.lower().endswith(".csv")
        doc_name = os.path.basename(file_path)

        juris_label = (account_country or "N/A").upper()
        prov_label = (provider or "N/A").upper()

        print("\n" + "=" * 80)
        print(f"📄 [{idx}/{total_docs}] Processing {'[CSV]' if is_csv else '[PDF]'}: \033[1m{doc_name}\033[0m")
        print(f"   Jurisdiction: {juris_label} | Provider: {prov_label}")
        print("=" * 80)

        if is_csv:
            file_sha = calculate_sha256(file_path)

            if db.is_source_document_ingested(file_sha) and not force_flag:
                print(f"  - ⏩ **Skipped [{idx}/{total_docs}]**: Already ingested.")
                continue

            try:
                staged_records = parse_directa_csv(file_path)
                inserted_records: list[StagedFinancialRecord] = []
                for stg in staged_records:
                    ins = db.insert_staged_record(stg)
                    inserted_records.append(ins)

                with Session(db.engine) as session:
                    doc_rec = IngestedSourceDocument(
                        file_sha=file_sha,
                        file_name=doc_name,
                        provider=provider or "",
                        account_country=account_country or "",
                        status="SUCCESS",
                        transaction_count=len(inserted_records),
                    )
                    session.add(doc_rec)
                    session.commit()

                print(
                    f"\033[32;1m  - ✅ Successfully parsed & staged [{idx}/{total_docs}] ({doc_name}): {len(inserted_records)} transaction(s) detected.\033[0m"
                )
            except Exception as csv_err:
                print(f"❌ Error processing CSV file '{doc_name}': {csv_err}", file=sys.stderr)
        else:
            doc = IngestionDocument.from_file(
                file_path=file_path,
                account_country=account_country,
                provider=provider,
            )
            status, records = pipeline.ingest_records_document(
                doc=doc,
                force=force_flag,
                force_pii_reprocessing=force_pii_flag,
                parser=parser_str,
                force_ocr=force_ocr_flag,
            )
            if status == VerificationStatus.SKIPPED:
                print(f"  - ⏩ **Skipped [{idx}/{total_docs}]**: Already ingested.")
            elif status == VerificationStatus.ESCALATED_TO_USER:
                print(
                    f"  - ⚠️  **Escalated [{idx}/{total_docs}]** ({doc.doc_name}): Contains voter/FX disagreements! Use UI to review them."
                )
            else:
                print(
                    f"\033[32;1m  - ✅ Successfully processed [{idx}/{total_docs}] ({doc.doc_name}): {len(records)} transaction(s) detected & staged.\033[0m"
                )

    print("\n" + "=" * 80)
    print("\033[32;1m🎉 Batch Ingestion execution completed successfully!\033[0m")
    print("=" * 80 + "\n")


def main() -> None:
    """Main CLI entry point for transaction ingestion."""
    log_env_vars(logging.getLogger(__name__))

    parser = _build_arg_parser()
    args = parser.parse_args()

    command: str = str(args.command)

    db = DatabaseManager(db_config=LocalDb(db_path=args.db))

    try:
        if command == "list":
            _handle_list(db)
            return

        if command == "delete":
            if getattr(args, "delete_record", None) is not None:
                _handle_delete_record(db, int(args.delete_record))
            elif getattr(args, "delete_all", False):
                _handle_delete_all(db)
            return

        if command == "ingest":
            # Default command: ingest
            mode_val = getattr(args, "mode", "api")
            test_val = getattr(args, "test", False)
            if mode_val == "mock" and not test_val:
                print(
                    f"❌ Error: Running in non-production mode ('{mode_val}') requires explicit '--test' flag.",
                    file=sys.stderr,
                )
                print(
                    f"   Usage: python backend/ingestion/ingest_transactions.py ingest --mode {mode_val} --test",
                    file=sys.stderr,
                )
                sys.exit(1)

            _handle_batch_ingestion(db, args)
            return

        print("Usage: python backend/ingestion/ingest_transactions.py [list|ingest|delete] ...")

    finally:
        db.close()


if __name__ == "__main__":
    main()
