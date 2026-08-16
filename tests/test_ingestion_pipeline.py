import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from backend.consensus_models import TransactionExtractionItem
from backend.db_manager import DatabaseManager, LocalDb
from backend.domain_models import AssetType, TransactionAction
from backend.ingestion.helpers import IngestionDocument
from backend.ingestion.parser import ParsedPage
from backend.ingestion.pii.models import PIIPipelineConfig
from backend.ingestion.pipeline import ConsensusVerifier, PIIPipeline, TransactionPipeline, VerificationStatus
from backend.llm.runner import BaseLLMRunner

APPROVED_ITEM: dict[str, object] = {
    "event_date": "2025-06-15T12:00:00",
    "asset_type": "stock",
    "symbol": "AAPL",
    "action": "buy",
    "quantity": 10.0,
    "unit_price": 180.0,
    "currency": "EUR",
    "fees": 1.5,
    "total_amount": 1801.5,
    "fx_rate": 1.0,
}


class ScriptedRunner(BaseLLMRunner):
    """Return scripted JSON responses for deterministic pipeline tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    @property
    def model_name(self) -> str:
        """Return test runner name."""
        return "scripted-test-runner"

    def complete(self, prompt: str, system_instruction: str) -> str:
        """Return next configured response."""
        del prompt, system_instruction
        if not self.responses:
            raise AssertionError("Scripted runner exhausted")
        return self.responses.pop(0)


class FixtureParser:
    """Parser double returning fixture text without reading or parsing PDFs."""

    @classmethod
    def parse_pdf(cls, file_path: str, force_parsing: bool = False) -> list[ParsedPage]:
        """Return one static parsed page."""
        del file_path, force_parsing
        return [ParsedPage(page_number=1, combined_content="fixture transaction text")]


def _transaction_json(item: Mapping[str, object]) -> str:
    return json.dumps([item])


def _document(tmp_path: Path, *, account_country: str | None = "italy") -> IngestionDocument:
    source = tmp_path / "fixture.pdf"
    source.write_bytes(b"fixture bytes; parser must not read this")
    return IngestionDocument.from_file(str(source), account_country=account_country, provider="fixture-provider")


@pytest.fixture
def db(tmp_path: Path):
    """Provide isolated database, closed after each test."""
    database = DatabaseManager(db_config=LocalDb(db_path=str(tmp_path / "test.db"), vector_db_path=str(tmp_path / "vector.db")))
    yield database
    database.close()


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, db: DatabaseManager) -> TransactionPipeline:
    """Build pipeline with parser registry replaced by fixture parser."""
    monkeypatch.setattr("backend.ingestion.pipeline.get_parser_registry", lambda: {"fixture": FixtureParser})
    pii = PIIPipeline(
        PIIPipelineConfig(
            presidio_enabled=True,
            openai_filter_enabled=False,
            llm_redaction=False,
            pii_cache_enabled=False,
        )
    )
    default_runner = ScriptedRunner([_transaction_json(APPROVED_ITEM)] * 3)
    return TransactionPipeline(db, [default_runner, default_runner, default_runner], pii, save_logs=False)


def _build_pipeline(db: DatabaseManager, runner: BaseLLMRunner, pii: PIIPipeline) -> TransactionPipeline:
    return TransactionPipeline(db, [runner, runner, runner], pii, save_logs=False)


def _pii_pipeline() -> PIIPipeline:
    return PIIPipeline(
        PIIPipelineConfig(
            presidio_enabled=True,
            openai_filter_enabled=False,
            llm_redaction=False,
            pii_cache_enabled=False,
        )
    )


def test_consensus_verifier_approves_identical_votes() -> None:
    item = TransactionExtractionItem(
        event_date=datetime(2025, 6, 15, 12),
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        action=TransactionAction.BUY,
        quantity=Decimal("10"),
        unit_price=Decimal("180"),
        currency="USD",
        fees=Decimal("1.5"),
        total_amount=Decimal("1801.5"),
        fx_rate=Decimal("0.92"),
    )

    result = ConsensusVerifier.verify_consensus([[item], [item.model_copy()], [item.model_copy()]])

    assert result.status == VerificationStatus.APPROVED
    assert result.candidate_records[0].symbol == "AAPL"


def test_consensus_verifier_escalates_count_mismatch() -> None:
    item = TransactionExtractionItem(
        event_date=datetime(2025, 6, 15, 12),
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        action=TransactionAction.BUY,
        quantity=Decimal("10"),
        unit_price=Decimal("180"),
        total_amount=Decimal("1800"),
    )

    result = ConsensusVerifier.verify_consensus([[item], [item, item.model_copy()], [item]])

    assert result.status == VerificationStatus.ESCALATED_TO_USER
    assert result.consensus_log.error == "Mismatch in transaction counts"


def test_consensus_verifier_escalates_value_mismatch() -> None:
    item = TransactionExtractionItem(
        event_date=datetime(2025, 6, 15, 12),
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        action=TransactionAction.BUY,
        quantity=Decimal("10"),
        unit_price=Decimal("180"),
        total_amount=Decimal("1800"),
    )
    mismatch = item.model_copy(update={"quantity": Decimal("10.5")})

    result = ConsensusVerifier.verify_consensus([[item], [item.model_copy()], [mismatch]])

    assert result.status == VerificationStatus.ESCALATED_TO_USER
    assert result.consensus_log.mismatches[0].voter3 is not None
    assert result.consensus_log.mismatches[0].voter3.quantity == Decimal("10.5")


def test_pipeline_persists_approved_fixture_records(
    pipeline: TransactionPipeline, db: DatabaseManager, tmp_path: Path
) -> None:
    runner = ScriptedRunner([_transaction_json(APPROVED_ITEM)] * 3)
    pipeline.extractor.runners = [runner, runner, runner]

    status, records = pipeline.ingest_records_document(_document(tmp_path), parser="fixture", force=True)

    assert status == VerificationStatus.APPROVED
    assert len(records) == 1
    assert records[0].symbol == "AAPL"
    assert records[0].verification_status == "pending_approval"
    staged = db.get_staged_records(account_country="italy")
    assert len(staged) == 1
    log_data = json.loads(staged[0].consensus_log or "{}")
    assert all(log_data.get(f"raw_voter_{index}_records") for index in (1, 2, 3))


def test_pipeline_escalates_and_persists_mismatch(
    pipeline: TransactionPipeline, db: DatabaseManager, tmp_path: Path
) -> None:
    mismatch = dict(APPROVED_ITEM, quantity=10.5)
    runner = ScriptedRunner(
        [_transaction_json(APPROVED_ITEM), _transaction_json(APPROVED_ITEM), _transaction_json(mismatch)]
    )
    pipeline.extractor.runners = [runner, runner, runner]

    status, records = pipeline.ingest_records_document(_document(tmp_path), parser="fixture", force=True)

    assert status == VerificationStatus.ESCALATED_TO_USER
    assert records[0].verification_status == "escalated_to_user"
    log_data = json.loads(db.get_staged_records(account_country="italy")[0].consensus_log or "{}")
    assert log_data["mismatches"]


def test_pipeline_rejects_unknown_parser(pipeline: TransactionPipeline, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown parser 'missing'"):
        pipeline.ingest_records_document(_document(tmp_path), parser="missing")


def test_pipeline_marks_failed_source_on_invalid_voter_response(
    pipeline: TransactionPipeline, db: DatabaseManager, tmp_path: Path
) -> None:
    runner = ScriptedRunner(["not JSON"] * 3)
    pipeline.extractor.runners = [runner, runner, runner]
    doc = _document(tmp_path)

    with pytest.raises(ValueError, match="failed to return a compliant JSON schema"):
        pipeline.ingest_records_document(doc, parser="fixture", force=True)

    tracking = db.get_ingested_source_document(doc.sha)
    assert tracking is not None
    assert tracking.status == "FAILED"


def test_pipeline_skips_successfully_ingested_document(
    db: DatabaseManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("backend.ingestion.pipeline.get_parser_registry", lambda: {"fixture": FixtureParser})
    runner = ScriptedRunner([_transaction_json(APPROVED_ITEM)] * 3)
    pipeline = _build_pipeline(db, runner, _pii_pipeline())
    doc = _document(tmp_path)

    first_status, _ = pipeline.ingest_records_document(doc, parser="fixture", force=True)
    second_status, _ = pipeline.ingest_records_document(doc, parser="fixture")

    assert first_status == VerificationStatus.APPROVED
    assert second_status == VerificationStatus.SKIPPED


def test_pipeline_requires_account_country(
    db: DatabaseManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("backend.ingestion.pipeline.get_parser_registry", lambda: {"fixture": FixtureParser})
    runner = ScriptedRunner([_transaction_json(APPROVED_ITEM)] * 3)
    pipeline = _build_pipeline(db, runner, _pii_pipeline())

    with pytest.raises(ValueError, match="missing mandatory .*account_country.*"):
        pipeline.ingest_records_document(_document(tmp_path, account_country=None), parser="fixture", force=True)
