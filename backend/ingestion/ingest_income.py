"""Command-Line Interface for Tax Income Document Ingestion Subsystem.

Provides CLI commands to ingest official employment/income tax documents (e.g. Irish EDS),
stage them with multi-voter LLM extraction, list staged and approved income records,
approve staged records into the official ledger, and delete records.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Annotated, Literal

import dotenv
from pydantic import Field, TypeAdapter

from backend.cli_common import (
    CommonConfigArgs,
    build_common_config_parser,
    parse_typed_args,
)
from backend.config import AppConfig
from backend.db_manager import DatabaseManager, LocalDb
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictStagedTaxIncomeRecord,
    StrictTaxIncomeRecord,
)
from backend.ingestion.helpers import calculate_sha256, log_env_vars
from backend.ingestion.income_pipeline import IncomeIngestionPipeline
from backend.ingestion.parser_registry import get_parser_registry
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.llm.runner_factory import build_pydantic_ai_runner

logger = logging.getLogger(__name__)

# Load environment variables
_ = dotenv.load_dotenv()


def print_staged_income_details(r: StrictStagedTaxIncomeRecord) -> None:
    """Print staged tax income record details in a clean tabular format."""
    payload = r.payload
    doc_name = r.source_file_name or "N/A"
    status_str = r.verification_status.upper()
    approved_id = str(r.approved_tax_income_record_id or "None")

    print("┌" + "─" * 78 + "┐")
    print(
        f"│ Staged ID: {r.id or 0:<8} │ Year: {r.tax_year:<6} │ "
        f"Jurisdiction: {r.jurisdiction:<12} │ Status: {status_str:<10} │"
    )
    print("├" + "─" * 78 + "┤")
    print(f"│ Source File: {doc_name:<40} │ Approved ID: {approved_id:<13} │")

    if isinstance(payload, IrishEmploymentDetailSummaryPayload):
        emp_name = payload.employer_name or "N/A"
        gross_str = f"€{payload.gross_pay_eur:,.2f}"
        tax_pay_str = f"€{payload.pay_for_income_tax_eur:,.2f}" if payload.pay_for_income_tax_eur is not None else "N/A"
        tax_str = f"€{payload.income_tax_paid_eur:,.2f}"
        bik_str = f"€{payload.taxable_benefits_eur:,.2f}" if payload.taxable_benefits_eur is not None else "N/A"
        usc_pay_str = f"€{payload.pay_for_usc_eur:,.2f}" if payload.pay_for_usc_eur is not None else "N/A"
        usc_str = f"€{payload.usc_paid_eur:,.2f}"
        prsi_str = f"€{payload.prsi_paid_eur:,.2f}"
        ern_str = payload.employer_registration_number or "N/A"
        weeks_str = str(payload.prsi_weeks) if payload.prsi_weeks is not None else "N/A"
        prsi_class_str = str(payload.prsi_class or "N/A")

        if payload.prsi_classes:
            classes_summary = ", ".join(f"{c.prsi_class} ({c.insurable_weeks}w)" for c in payload.prsi_classes)
        else:
            classes_summary = f"{prsi_class_str} ({weeks_str}w)"

        print(f"│ Employer: {emp_name:<42} │ ERN: {ern_str:<18} │")
        print(f"│ Gross Pay: {gross_str:<16} │ Pay for Tax: {tax_pay_str:<14} │ Tax Paid: {tax_str:<12} │")
        print(f"│ Taxable BIK: {bik_str:<14} │ Pay for USC: {usc_pay_str:<14} │ USC Paid: {usc_str:<12} │")
        print(f"│ Employee PRSI: {prsi_str:<12} │ PRSI Classes: {classes_summary:<35} │")
    else:
        print("│ ⚠️ Status: Unresolved Discrepancies (Review in UI Voter Diff to resolve)     │")

    if r.discrepancies:
        print("├" + "─" * 78 + "┤")
        print("│ ⚠️ Discrepancies Detected:                                                  │")
        for d in r.discrepancies:
            print(f"│   - {d:<72} │")
    print("└" + "─" * 78 + "┘")


def print_approved_income_details(r: StrictTaxIncomeRecord) -> None:
    """Print approved tax income record details in a clean tabular format."""
    payload = r.payload
    created_str = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "N/A"

    print("┌" + "─" * 78 + "┐")
    print(
        f"│ Ledger ID: {r.id or 0:<8} │ Year: {r.tax_year:<6} │ "
        f"Jurisdiction: {r.jurisdiction:<12} │ Created: {created_str:<11} │"
    )
    print("├" + "─" * 78 + "┤")

    if isinstance(payload, IrishEmploymentDetailSummaryPayload):
        emp_name = payload.employer_name or "N/A"
        gross_str = f"€{payload.gross_pay_eur:,.2f}"
        tax_pay_str = f"€{payload.pay_for_income_tax_eur:,.2f}" if payload.pay_for_income_tax_eur is not None else "N/A"
        tax_str = f"€{payload.income_tax_paid_eur:,.2f}"
        bik_str = f"€{payload.taxable_benefits_eur:,.2f}" if payload.taxable_benefits_eur is not None else "N/A"
        usc_pay_str = f"€{payload.pay_for_usc_eur:,.2f}" if payload.pay_for_usc_eur is not None else "N/A"
        usc_str = f"€{payload.usc_paid_eur:,.2f}"
        prsi_str = f"€{payload.prsi_paid_eur:,.2f}"
        empr_prsi_str = str(payload.employer_prsi_paid_eur or "N/A")
        ern_str = payload.employer_registration_number or "N/A"
        weeks_str = str(payload.prsi_weeks) if payload.prsi_weeks is not None else "N/A"
        prsi_class_str = str(payload.prsi_class or "N/A")

        if payload.prsi_classes:
            classes_summary = ", ".join(f"{c.prsi_class} ({c.insurable_weeks}w)" for c in payload.prsi_classes)
        else:
            classes_summary = f"{prsi_class_str} ({weeks_str}w)"

        print(f"│ Employer: {emp_name:<42} │ ERN: {ern_str:<18} │")
        print(f"│ Gross Pay: {gross_str:<16} │ Pay for Tax: {tax_pay_str:<14} │ Tax Paid: {tax_str:<12} │")
        print(f"│ Taxable BIK: {bik_str:<14} │ Pay for USC: {usc_pay_str:<14} │ USC Paid: {usc_str:<12} │")
        print(
            f"│ Employee PRSI: {prsi_str:<12} │ Employer PRSI: {empr_prsi_str:<12} │ Classes: {classes_summary:<14} │"
        )

    print("└" + "─" * 78 + "┘")


class IngestSubcommandArgs(CommonConfigArgs):
    """Arguments for 'ingest' subcommand."""

    command: Literal["ingest"]
    force: bool = False
    force_pii_reprocessing: bool = False
    parser: str = "chandra"
    force_ocr: bool = False
    file: str | None = None
    folder: str | None = None
    path: str | None = None


class ListSubcommandArgs(CommonConfigArgs):
    """Arguments for 'list' subcommand."""

    command: Literal["list"]
    tax_year: int | None = None
    jurisdiction: str | None = None
    status: str | None = None
    target: Literal["staged", "approved", "all"] = "all"


class DeleteSubcommandArgs(CommonConfigArgs):
    """Arguments for 'delete' subcommand."""

    command: Literal["delete"]
    delete_record: int | None = None
    delete_all: bool = False
    staged: bool = False


IncomeCliArgs = Annotated[
    IngestSubcommandArgs | ListSubcommandArgs | DeleteSubcommandArgs,
    Field(discriminator="command"),
]
INCOME_CLI_ADAPTER: TypeAdapter[IncomeCliArgs] = TypeAdapter(IncomeCliArgs)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for income ingestion CLI."""
    parent_parser = build_common_config_parser()

    parser = argparse.ArgumentParser(
        description="Tax Income Ingestion Subsystem CLI (Multi-Voter Consensus + Staging)",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available subcommands ('ingest', 'list', 'approve', 'delete')",
    )

    # 1. Ingest
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest and stage PDF employment income summaries (e.g. Irish EDS)",
        parents=[parent_parser],
    )
    _ = p_ingest.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of already staged or approved documents",
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
        help="Path to a single PDF income document to ingest",
    )
    _ = target_group.add_argument(
        "--folder",
        "--dir",
        type=str,
        default=None,
        help="Path to a folder containing income documents to ingest",
    )
    _ = target_group.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Convenience alias: path to either a single file or a directory of documents to ingest",
    )

    # 2. List
    p_list = subparsers.add_parser(
        "list",
        help="List staged and/or approved tax income records",
        parents=[parent_parser],
    )
    _ = p_list.add_argument(
        "--tax-year",
        "-y",
        type=int,
        default=None,
        help="Filter by tax year integer (e.g. 2024, 2025)",
    )
    _ = p_list.add_argument(
        "--jurisdiction",
        "-j",
        type=str,
        default=None,
        help="Filter by jurisdiction ('ireland', 'italy')",
    )
    _ = p_list.add_argument(
        "--status",
        "-s",
        type=str,
        default=None,
        help="Filter by verification status ('auto_approved', 'majority_agreed', 'escalated_to_user')",
    )
    _ = p_list.add_argument(
        "--target",
        type=str,
        choices=["staged", "approved", "all"],
        default="all",
        help="Target table: 'staged', 'approved', or 'all' (default: 'all')",
    )

    # 3. Delete
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete tax income records from database",
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
        help="Delete all income records",
    )
    _ = p_delete.add_argument(
        "--staged",
        action="store_true",
        help="Target staged records (default targets approved records when --id or --all is used)",
    )

    return parser


def _collect_target_income_files(target_path: str) -> list[str]:
    """Collect all PDF files from target path for income document ingestion."""
    if os.path.isfile(target_path):
        return [target_path] if target_path.lower().endswith(".pdf") else []

    pdfs: list[str] = []
    for root, _, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, file))
    return pdfs


def _handle_ingestion(
    db: DatabaseManager,
    args: IngestSubcommandArgs,
    app_config: AppConfig,
) -> None:
    """Execute batch income document ingestion."""
    print("\n" + "=" * 80)
    print("📋 TAX INCOME INGESTION PIPELINE")
    print("=" * 80)

    # Initialize Voter Runners
    runners = [
        build_pydantic_ai_runner("VOTER_1"),
        build_pydantic_ai_runner("VOTER_2"),
        build_pydantic_ai_runner("VOTER_3"),
    ]
    print("🏛️  Voter Consensus Configuration:")
    for idx, r in enumerate(runners, start=1):
        print(f"   ├─ Voter {idx}: {r.model_name}")

    parser_cls = get_parser_registry()[args.parser]
    pii_pipeline = PIIPipeline()
    pipeline = IncomeIngestionPipeline(
        db=db,
        pii_pipeline=pii_pipeline,
        ocr_parser=parser_cls,
        runners=runners,
    )

    raw_target = args.file or args.folder or args.path
    target_path = str(raw_target) if raw_target else str(app_config.raw_records_dir)

    if not os.path.exists(target_path):
        print(f"Error: Specified path '{target_path}' not found.", file=sys.stderr)
        return

    files_to_process = _collect_target_income_files(target_path)
    if not files_to_process:
        print("📁 No matching PDF files found to ingest.")
        return

    total = len(files_to_process)
    print(f"\n📁 Found {total} income document(s) to process.\n")

    for idx, f_path in enumerate(files_to_process, start=1):
        doc_name = os.path.basename(f_path)
        doc_sha = calculate_sha256(f_path)

        print("-" * 80)
        print(f"📄 [{idx}/{total}] Processing: \033[1m{doc_name}\033[0m")

        if db.is_tax_income_document_ingested(doc_sha) and not args.force:
            print(f"   ⏩ \033[33mSkipped [{idx}/{total}]\033[0m: Already ingested.")
            continue

        if args.force:
            deleted_prev = db.delete_tax_income_records_by_sha(doc_sha)
            if deleted_prev > 0:
                print(f"   ↺ Cleared {deleted_prev} previously stored record(s) for document.")

        try:
            result = pipeline.ingest_irish_eds(
                file_path=f_path,
                force_ocr=args.force_ocr,
                force_pii=args.force_pii_reprocessing,
            )

            if result.status == "approved":
                rec_id = result.staged_record.id
                status = result.staged_record.verification_status
                print(f"   ✅ \033[32;1mSuccessfully staged record #{rec_id} ({status})\033[0m")
                print_staged_income_details(result.staged_record)
            elif result.status == "escalated":
                rec_id = result.staged_record.id
                print(f"   ⚠️  \033[33;1mEscalated record #{rec_id} (Discrepancies detected, review in UI)\033[0m")
                print_staged_income_details(result.staged_record)
            else:
                print(f"   ❌ \033[31;1mIngestion Failed:\033[0m {result.error_message}")
        except Exception as e:
            print(f"   ❌ Error ingesting '{doc_name}': {e}", file=sys.stderr)

    print("\n" + "=" * 80)
    print("\033[32;1m🎉 Income document batch ingestion completed successfully!\033[0m")
    print("=" * 80 + "\n")


def _handle_list(db: DatabaseManager, args: ListSubcommandArgs) -> None:
    """Handle listing staged and/or approved records."""
    if args.target in ("staged", "all"):
        staged = db.get_staged_tax_income_records(
            tax_year=args.tax_year,
            jurisdiction=args.jurisdiction,
            status=args.status,
        )
        print(f"\n📂 Staged Tax Income Records ({len(staged)}):")
        for s in staged:
            print_staged_income_details(s)

    if args.target in ("approved", "all"):
        approved = db.get_tax_income_records(
            tax_year=args.tax_year,
            jurisdiction=args.jurisdiction,
        )
        print(f"\n📑 Approved Tax Income Records ({len(approved)}):")
        for a in approved:
            print_approved_income_details(a)


def _handle_delete_staged(db: DatabaseManager, record_id: int | None, delete_all: bool) -> None:
    """Handle deletion of staged income records."""
    if record_id is not None:
        staged = db.get_staged_tax_income_record_by_id(record_id)
        if not staged:
            print(f"❌ Staged record #{record_id} not found.")
            return
        print_staged_income_details(staged)
        confirm = input(f"Delete staged record #{record_id}? [y/N]: ").strip().lower()
        if confirm == "y":
            db.delete_staged_tax_income_record(record_id)
            print(f"✅ Staged record #{record_id} deleted.")
        else:
            print("Aborted.")
    elif delete_all:
        confirm = input("Are you sure you want to delete ALL staged income records? [y/N]: ").strip().lower()
        if confirm == "y":
            count = db.delete_all_staged_tax_income_records()
            print(f"✅ Deleted {count} staged record(s).")
        else:
            print("Aborted.")


def _handle_delete_approved(db: DatabaseManager, record_id: int | None, delete_all: bool) -> None:
    """Handle deletion of approved income records."""
    if record_id is not None:
        confirm = input(f"Delete approved record #{record_id}? [y/N]: ").strip().lower()
        if confirm == "y":
            deleted = db.delete_tax_income_record(record_id)
            if deleted:
                print(f"✅ Approved record #{record_id} deleted.")
            else:
                print(f"❌ Record #{record_id} not found.")
        else:
            print("Aborted.")
    elif delete_all:
        confirm = input("Are you sure you want to delete ALL approved income records? [y/N]: ").strip().lower()
        if confirm == "y":
            count = db.delete_all_tax_income_records()
            print(f"✅ Deleted {count} approved record(s).")
        else:
            print("Aborted.")


def _handle_delete(db: DatabaseManager, args: DeleteSubcommandArgs) -> None:
    """Dispatch deletion for staged or approved records."""
    if args.staged:
        _handle_delete_staged(db, args.delete_record, args.delete_all)
    else:
        _handle_delete_approved(db, args.delete_record, args.delete_all)


def main() -> None:
    """Main CLI entry point for income ingestion."""
    log_env_vars(logging.getLogger(__name__))

    parser = _build_arg_parser()
    args: IncomeCliArgs = parse_typed_args(parser, INCOME_CLI_ADAPTER)
    app_config = args.resolve_app_config()

    db = DatabaseManager(
        db_config=LocalDb(
            db_path=app_config.db_path,
            vector_db_path=app_config.vector_db_path,
        )
    )

    try:
        if args.command == "ingest":
            _handle_ingestion(db, args, app_config)
        elif args.command == "list":
            _handle_list(db, args)
        elif args.command == "delete":
            _handle_delete(db, args)
    finally:
        db.close()


if __name__ == "__main__":
    main()
