"""Tests for Tax Income Ingestion Subsystem CLI (ingest_income.py)."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from backend.db_manager import DatabaseManager, MemoryDb
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictStagedTaxIncomeRecord,
    StrictTaxIncomeRecord,
)
from backend.ingestion.ingest_income import (
    INCOME_CLI_ADAPTER,
    DeleteSubcommandArgs,
    IngestSubcommandArgs,
    ListSubcommandArgs,
    _build_arg_parser,
    _collect_target_income_files,
    _handle_list,
)


@pytest.fixture
def test_db() -> DatabaseManager:
    """Provide fresh in-memory database with tables initialized."""
    return DatabaseManager(MemoryDb())


def test_cli_arg_parser_ingest_options() -> None:
    # Given: Ingest subcommand CLI arguments
    parser = _build_arg_parser()
    raw_args = ["ingest", "--file", "sample_eds.pdf", "--force", "--force-ocr", "--parser", "pdfplumber"]

    # When: Parsing and validating through typed adapter
    parsed_ns = parser.parse_args(raw_args)
    typed_args = INCOME_CLI_ADAPTER.validate_python(vars(parsed_ns))

    # Then: Arguments correctly parsed into IngestSubcommandArgs
    assert isinstance(typed_args, IngestSubcommandArgs)
    assert typed_args.command == "ingest"
    assert typed_args.file == "sample_eds.pdf"
    assert typed_args.force is True
    assert typed_args.force_ocr is True
    assert typed_args.parser == "pdfplumber"


def test_cli_arg_parser_list_and_delete_options() -> None:
    # Given: List and Delete arguments
    parser = _build_arg_parser()

    # When: Parsing list
    list_ns = parser.parse_args(["list", "--tax-year", "2025", "--jurisdiction", "ireland", "--target", "staged"])
    list_args = INCOME_CLI_ADAPTER.validate_python(vars(list_ns))

    # Then: List options parsed
    assert isinstance(list_args, ListSubcommandArgs)
    assert list_args.tax_year == 2025
    assert list_args.jurisdiction == "ireland"
    assert list_args.target == "staged"

    # When: Parsing delete
    del_ns = parser.parse_args(["delete", "--id", "12", "--staged"])
    del_args = INCOME_CLI_ADAPTER.validate_python(vars(del_ns))

    # Then: Delete options parsed
    assert isinstance(del_args, DeleteSubcommandArgs)
    assert del_args.delete_record == 12
    assert del_args.staged is True


def test_collect_target_income_files(tmp_path: Path) -> None:
    # Given: Directory with mixed PDF and non-PDF files
    p1 = tmp_path / "eds_2024.pdf"
    p2 = tmp_path / "eds_2025.PDF"
    txt = tmp_path / "notes.txt"
    p1.write_text("dummy", encoding="utf-8")
    p2.write_text("dummy", encoding="utf-8")
    txt.write_text("dummy", encoding="utf-8")

    # When: Collecting target files
    collected = _collect_target_income_files(str(tmp_path))

    # Then: Only PDFs collected
    assert len(collected) == 2
    assert str(p1) in collected
    assert str(p2) in collected


def test_handle_list_and_delete(test_db: DatabaseManager, capsys: pytest.CaptureFixture[str]) -> None:
    # Given: Persisted staged and approved records
    payload = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Stripe Technology",
        gross_pay_eur=Decimal("80000.00"),
        income_tax_paid_eur=Decimal("20000.00"),
        usc_paid_eur=Decimal("3000.00"),
        prsi_paid_eur=Decimal("3200.00"),
    )
    staged = StrictStagedTaxIncomeRecord(
        tax_year=2025,
        jurisdiction="ireland",
        income_type=payload.income_type,
        payload=payload,
        verification_status="auto_approved",
        created_at=datetime.now(timezone.utc),
    )
    staged_id = test_db.insert_staged_tax_income_record(staged)

    approved = StrictTaxIncomeRecord(
        tax_year=2025,
        jurisdiction="ireland",
        income_type=payload.income_type,
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    app_id = test_db.insert_tax_income_record(approved)

    # When: Calling _handle_list
    list_args = ListSubcommandArgs(command="list", target="all")
    _handle_list(test_db, list_args)

    # Then: Output captures both records
    captured = capsys.readouterr().out
    assert "Staged Tax Income Records" in captured
    assert "Approved Tax Income Records" in captured
    assert "Stripe Technology" in captured
    assert "€80,000.00" in captured

    # When: Calling delete on approved record via helper
    test_db.delete_tax_income_record(app_id)
    test_db.delete_staged_tax_income_record(staged_id)

    assert len(test_db.get_tax_income_records()) == 0
    assert len(test_db.get_staged_tax_income_records()) == 0
