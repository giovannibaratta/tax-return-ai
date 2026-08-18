"""Tests for Generalized Income Records Ingestion, Multi-Voter Consensus, and Tax Tools."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from backend.db_manager import DatabaseManager, MemoryDb
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictTaxIncomeRecord,
)
from backend.ingestion.generic_voter import (
    default_reconciler,
    run_multi_voter_consensus,
)
from backend.ingestion.income_pipeline import (
    IncomeIngestionPipeline,
    IngestionEscalated,
    IngestionSuccess,
)
from backend.ingestion.parser import BasePDFParser, ParsedPage
from backend.ingestion.pii.pii_pipeline import PIIPipeline, PIIPipelineConfig
from backend.llm.pydantic_ai_runner import PydanticAIRunner
from backend.services.tax_services import get_tax_income_records_action
from tests.utils import DummyMockPydanticRunner


@pytest.fixture
def test_db() -> DatabaseManager:
    """Provide fresh in-memory database with applied migrations."""
    return DatabaseManager(MemoryDb())


def test_generic_voter_consensus_identical() -> None:
    # Given: 3 runners returning identical model payload
    payload = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Acme Corp",
        gross_pay_eur=Decimal("60000.00"),
        income_tax_paid_eur=Decimal("15000.00"),
        usc_paid_eur=Decimal("2400.00"),
        prsi_paid_eur=Decimal("2400.00"),
    )
    runners: list[PydanticAIRunner] = [
        DummyMockPydanticRunner(payload),
        DummyMockPydanticRunner(payload),
        DummyMockPydanticRunner(payload),
    ]

    # When: running generic consensus
    result = run_multi_voter_consensus(
        prompt="Extract EDS",
        system_prompt="System prompt",
        schema_cls=IrishEmploymentDetailSummaryPayload,
        runners=runners,
    )

    # Then: approved without discrepancies
    assert result.status == "approved"
    assert result.reconciled_output is not None
    assert result.reconciled_output.employer_name == "Acme Corp"
    assert result.reconciled_output.gross_pay_eur == Decimal("60000.00")
    assert len(result.discrepancies) == 0


def test_generic_voter_consensus_majority() -> None:
    # Given: 2/3 majority agreement on gross_pay
    p1 = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Acme Corp",
        gross_pay_eur=Decimal("60000.00"),
        income_tax_paid_eur=Decimal("15000.00"),
        usc_paid_eur=Decimal("2400.00"),
        prsi_paid_eur=Decimal("2400.00"),
    )
    p2 = p1.model_copy()
    p3 = p1.model_copy(update={"gross_pay_eur": Decimal("65000.00")})
    runners: list[PydanticAIRunner] = [
        DummyMockPydanticRunner(p1),
        DummyMockPydanticRunner(p2),
        DummyMockPydanticRunner(p3),
    ]

    # When: running generic consensus
    result = run_multi_voter_consensus(
        prompt="Extract EDS",
        system_prompt="System prompt",
        schema_cls=IrishEmploymentDetailSummaryPayload,
        runners=runners,
    )

    # Then: approved via majority consensus
    assert result.status == "approved"
    assert result.reconciled_output is not None
    assert result.reconciled_output.gross_pay_eur == Decimal("60000.00")
    assert len(result.discrepancies) == 1
    assert "gross_pay_eur" in result.discrepancies[0]


@pytest.mark.anyio
async def test_get_tax_income_records_tool(test_db: DatabaseManager) -> None:
    # Given: Persisted tax income record in database
    payload = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Meta Platforms Ireland",
        gross_pay_eur=Decimal("110000.00"),
        income_tax_paid_eur=Decimal("35000.00"),
        usc_paid_eur=Decimal("5000.00"),
        prsi_paid_eur=Decimal("4400.00"),
    )
    strict_record = StrictTaxIncomeRecord(
        tax_year=2025,
        jurisdiction="ireland",
        income_type=payload.income_type,
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    _ = test_db.insert_tax_income_record(strict_record)

    # When: Calling get_tax_income_records tool action
    records = get_tax_income_records_action(db=test_db, tax_year=2025, jurisdiction="ireland")

    # Then: Retrieved properly
    assert len(records) == 1
    assert records[0].payload.gross_pay_eur == Decimal("110000.00")
    assert records[0].payload.employer_name == "Meta Platforms Ireland"


class InnerModel(BaseModel):
    key: str
    amount: Decimal


class OuterModel(BaseModel):
    name: str
    inner: InnerModel
    tags: list[str]


def test_default_reconciler_nested_and_lists() -> None:
    # Given: 3 models with nested structures and 2/3 majority on nested amount
    m1 = OuterModel(name="Alpha", inner=InnerModel(key="K1", amount=Decimal("100")), tags=["t1", "t2"])
    m2 = OuterModel(name="Alpha", inner=InnerModel(key="K1", amount=Decimal("100")), tags=["t1", "t2"])
    m3 = OuterModel(name="Alpha", inner=InnerModel(key="K1", amount=Decimal("150")), tags=["t1", "t2"])

    # When: Reconciling
    reconciled, disc = default_reconciler([m1, m2, m3])

    # Then: Nested field reconciled via majority
    assert reconciled is not None
    assert reconciled.inner.amount == Decimal("100")
    assert reconciled.tags == ["t1", "t2"]
    assert len(disc) == 1
    assert "Nested 'inner'" in disc[0]


def test_default_reconciler_income_summary_exact_match() -> None:
    # Given: 3 identical Irish EDS outputs with all optional fields filled
    dt_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    dt_end = datetime(2025, 12, 31, tzinfo=timezone.utc)
    eds1 = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Stripe Technology Europe",
        employer_registration_number="9876543A",
        employment_id="1",
        start_date=dt_start,
        end_date=dt_end,
        gross_pay_eur=Decimal("120000.00"),
        income_tax_paid_eur=Decimal("38000.00"),
        usc_paid_eur=Decimal("5400.00"),
        prsi_paid_eur=Decimal("4800.00"),
        employer_prsi_paid_eur=Decimal("13260.00"),
        prsi_class="A1",
        prsi_weeks=52,
        lpt_deducted_eur=Decimal("450.00"),
    )
    eds2 = eds1.model_copy()
    eds3 = eds1.model_copy()

    # When: Reconciling
    reconciled, disc = default_reconciler([eds1, eds2, eds3])

    # Then: Cleanly reconciled with 0 discrepancies
    assert reconciled is not None
    assert reconciled == eds1
    assert len(disc) == 0


def test_default_reconciler_income_summary_majority_fields() -> None:
    # Given: 3 EDS outputs where voter 3 differs on PRSI weeks and tax paid
    eds1 = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Apple Distribution International",
        employer_registration_number="1234567T",
        gross_pay_eur=Decimal("95000.00"),
        income_tax_paid_eur=Decimal("28000.00"),
        usc_paid_eur=Decimal("4200.00"),
        prsi_paid_eur=Decimal("3800.00"),
        prsi_weeks=52,
    )
    eds2 = eds1.model_copy()
    eds3 = eds1.model_copy(
        update={
            "income_tax_paid_eur": Decimal("28500.00"),
            "prsi_weeks": 50,
        }
    )

    # When: Reconciling
    reconciled, disc = default_reconciler([eds1, eds2, eds3])

    # Then: Reconciled to majority values (28000.00 tax and 52 weeks)
    assert reconciled is not None
    assert reconciled.income_tax_paid_eur == Decimal("28000.00")
    assert reconciled.prsi_weeks == 52
    assert len(disc) == 2
    assert any("income_tax_paid_eur" in d for d in disc)
    assert any("prsi_weeks" in d for d in disc)


def test_default_reconciler_income_summary_irreconcilable_mismatch() -> None:
    # Given: 3 voters returning 3 distinct gross pay amounts (no majority)
    base = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="TikTok Technology Ltd",
        gross_pay_eur=Decimal("80000.00"),
        income_tax_paid_eur=Decimal("20000.00"),
        usc_paid_eur=Decimal("3000.00"),
        prsi_paid_eur=Decimal("3200.00"),
    )
    eds1 = base.model_copy(update={"gross_pay_eur": Decimal("80000.00")})
    eds2 = base.model_copy(update={"gross_pay_eur": Decimal("82000.00")})
    eds3 = base.model_copy(update={"gross_pay_eur": Decimal("84000.00")})

    # When: Reconciling
    reconciled, disc = default_reconciler([eds1, eds2, eds3])

    # Then: Fails reconciliation and reports no majority
    assert reconciled is None
    assert len(disc) == 1
    assert "gross_pay_eur" in disc[0]
    assert "no majority consensus" in disc[0]


def test_default_reconciler_edge_cases() -> None:
    # Given: Empty outputs list
    r_empty, disc_empty = default_reconciler([])
    assert r_empty is None
    assert len(disc_empty) == 1

    # Given: Single output
    single = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Single Employer",
        gross_pay_eur=Decimal("50000.00"),
        income_tax_paid_eur=Decimal("10000.00"),
        usc_paid_eur=Decimal("2000.00"),
        prsi_paid_eur=Decimal("2000.00"),
    )
    r_single, disc_single = default_reconciler([single])
    assert r_single == single
    assert len(disc_single) == 0


def test_income_ingestion_pipeline_success_and_escalated(test_db: DatabaseManager, tmp_path: Path) -> None:
    # Given: Dummy PDF file and mock parser
    dummy_pdf = tmp_path / "eds_sample.pdf"
    dummy_pdf.write_text("dummy pdf binary content", encoding="utf-8")

    class MockParser(BasePDFParser):
        @classmethod
        def parse_pdf(cls, file_path: str, force_parsing: bool = False, **kwargs: object) -> list[ParsedPage]:
            return [ParsedPage(page_number=1, combined_content="Employment Detail Summary 2025 Acme Corp")]

    # 1. Test IngestionSuccess
    payload_agree = IrishEmploymentDetailSummaryPayload(
        tax_year=2025,
        employer_name="Acme Corp",
        gross_pay_eur=Decimal("60000.00"),
        income_tax_paid_eur=Decimal("15000.00"),
        usc_paid_eur=Decimal("2400.00"),
        prsi_paid_eur=Decimal("2400.00"),
    )
    cfg = PIIPipelineConfig(presidio_enabled=True, llm_redaction=False, pii_cache_enabled=False)
    pipeline_success = IncomeIngestionPipeline(
        db=test_db,
        pii_pipeline=PIIPipeline(config=cfg),
        ocr_parser=MockParser,
        runners=[DummyMockPydanticRunner(payload_agree)],
    )

    # When: ingesting with unanimous voters
    res_success = pipeline_success.ingest_irish_eds(str(dummy_pdf))

    # Then: returns IngestionSuccess with staged record
    assert isinstance(res_success, IngestionSuccess)
    assert res_success.status == "approved"
    assert res_success.staged_record.tax_year == 2025
    assert res_success.staged_record.payload is not None
    assert res_success.staged_record.payload.employer_name == "Acme Corp"
    assert res_success.staged_record.verification_status == "auto_approved"
    assert res_success.staged_record.id is not None


    # And: record can be promoted to approved ledger
    approved_id = test_db.approve_staged_tax_income_record(res_success.staged_record.id)
    assert approved_id is not None
    approved_list = test_db.get_tax_income_records(tax_year=2025)
    assert len(approved_list) == 1
    assert approved_list[0].id == approved_id

    # 2. Test IngestionEscalated
    p1 = payload_agree.model_copy(update={"gross_pay_eur": Decimal("60000.00")})
    p2 = payload_agree.model_copy(update={"gross_pay_eur": Decimal("65000.00")})
    p3 = payload_agree.model_copy(update={"gross_pay_eur": Decimal("70000.00")})

    pipeline_escalated = IncomeIngestionPipeline(
        db=test_db,
        pii_pipeline=PIIPipeline(config=cfg),
        ocr_parser=MockParser,
        runners=[DummyMockPydanticRunner(p1), DummyMockPydanticRunner(p2), DummyMockPydanticRunner(p3)],
    )

    # When: ingesting with irreconcilable split voters
    res_escalated = pipeline_escalated.ingest_irish_eds(str(dummy_pdf))

    # Then: returns IngestionEscalated with staged record marked escalated_to_user and payload=None
    assert isinstance(res_escalated, IngestionEscalated)
    assert res_escalated.status == "escalated"
    assert res_escalated.staged_record.verification_status == "escalated_to_user"
    assert res_escalated.staged_record.payload is None
    assert len(res_escalated.consensus_result.discrepancies) > 0
