"""Pipeline for ingesting and verifying official tax income documents (e.g. Irish Revenue EDS).

Integrates:
1. OCR parsing via Chandra (local or remote Datalab API with disk caching).
2. PII sanitization via PIIPipeline.
3. Multi-voter extraction consensus via GenericVoter.
4. De-anonymization and persistence into tax_income_records.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from backend.db_manager import DatabaseManager
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictStagedTaxIncomeRecord,
)
from backend.ingestion.generic_voter import GenericConsensusResult, run_multi_voter_consensus
from backend.ingestion.helpers import calculate_sha256
from backend.ingestion.ocr_parser import ChandraAPIParser, ChandraOCRParser
from backend.ingestion.parser import BasePDFParser
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.llm.pydantic_ai_runner import PydanticAIRunner

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class IngestionSuccess(BaseModel, Generic[T]):
    """Successful income document ingestion with staged record and approved consensus."""

    status: Literal["approved"] = "approved"
    staged_record: StrictStagedTaxIncomeRecord
    consensus_result: GenericConsensusResult[T]


class IngestionEscalated(BaseModel, Generic[T]):
    """Escalated income document ingestion with staged record requiring manual review."""

    status: Literal["escalated"] = "escalated"
    staged_record: StrictStagedTaxIncomeRecord
    consensus_result: GenericConsensusResult[T]


class IngestionFailed(BaseModel, Generic[T]):
    """Failed income document ingestion due to unrecoverable extraction or voter failure."""

    status: Literal["failed"] = "failed"
    error_message: str
    staged_record: StrictStagedTaxIncomeRecord | None = None
    consensus_result: GenericConsensusResult[T] | None = None


IrishEDSIngestionResult = Annotated[
    IngestionSuccess[IrishEmploymentDetailSummaryPayload]
    | IngestionEscalated[IrishEmploymentDetailSummaryPayload]
    | IngestionFailed[IrishEmploymentDetailSummaryPayload],
    Field(discriminator="status"),
]

IRISH_EDS_SYSTEM_PROMPT = """You are an expert Irish tax accounting extraction engine.
Your task is to extract official employment income details from an Irish Revenue
"Employment Detail Summary" (EDS / P60 replacement) document into a single JSON object.


Extract the following fields strictly into the JSON schema:
- income_type: "irish_employment_detail_summary"
- tax_year: Tax year integer (e.g. 2024, 2025)
- employer_name: Full registered name of employer
- employer_registration_number: Employer tax registration number / ERN (or null)
- employment_id: Employment sequence number/ID (or null)
- start_date: ISO 8601 start date (YYYY-MM-DD) if available, else null
- end_date: ISO 8601 end date (YYYY-MM-DD) if available, else null
- gross_pay_eur: Total "Gross pay" in EUR (positive Decimal)
- pay_for_income_tax_eur: Total "Pay for Income Tax" in EUR (positive Decimal, or null)
- income_tax_paid_eur: Total "Income Tax paid" (PAYE) deducted in EUR (positive Decimal)
- taxable_benefits_eur: Total "Taxable benefits" in EUR (positive Decimal, or null)
- pay_for_usc_eur: Total "Pay for USC" in EUR (positive Decimal, or null)
- usc_paid_eur: Total "USC paid" (Universal Social Charge) deducted in EUR (positive Decimal)
- lpt_deducted_eur: "LPT deducted" (Local Property Tax) in EUR (positive Decimal, or null)
- prsi_paid_eur: "Employee PRSI paid" in EUR (positive Decimal)
- employer_prsi_paid_eur: "Employer PRSI paid" in EUR (positive Decimal, or null)
- prsi_classes: List of PRSI class entries [{"prsi_class": "A1", "insurable_weeks": 12}, ...]
  from the "PRSI classes" section
- prsi_class: Primary active PRSI class letter (e.g. "A1", or comma-separated if multiple with active weeks)
- prsi_weeks: Total number of insurable weeks summed across active PRSI classes (or null)


RULES:
1. Strip currency symbols (€, EUR) and format numbers with a decimal dot (e.g. 75000.50).
2. For multiple PRSI class entries (e.g. Class M: 0 weeks, Class A1: 12 weeks), populate each entry into `prsi_classes`.
3. Do not fabricate values. Use null for missing optional fields.
4. Return strictly valid JSON conforming to the schema.
"""


def _get_default_ocr_parser() -> type[BasePDFParser]:
    """Resolve default OCR parser: ChandraAPIParser if DATALAB_API_KEY set, else ChandraOCRParser."""
    if os.environ.get("DATALAB_API_KEY"):
        return ChandraAPIParser
    return ChandraOCRParser


class IncomeIngestionPipeline:
    """Manages end-to-end ingestion, consensus verification, and staging of tax income documents."""

    def __init__(
        self,
        db: DatabaseManager,
        pii_pipeline: PIIPipeline | None = None,
        ocr_parser: type[BasePDFParser] | None = None,
        runners: list[PydanticAIRunner] | None = None,
    ) -> None:
        """Initialize income ingestion pipeline.

        Args:
            db: DatabaseManager instance.
            pii_pipeline: Optional PIIPipeline instance (created by default if None).
            ocr_parser: Optional BasePDFParser class (defaults to Chandra parser).
            runners: Optional list of LLM runners for consensus voting.
        """
        self.db = db
        self.pii_pipeline = pii_pipeline or PIIPipeline()
        self.ocr_parser = ocr_parser or _get_default_ocr_parser()
        self.runners = runners

    def ingest_irish_eds(
        self,
        file_path: str,
        force_ocr: bool = False,
        force_pii: bool = False,
    ) -> IrishEDSIngestionResult:
        """Ingest and verify an Irish Revenue Employment Detail Summary (EDS) PDF.

        Tax year and employment fields are derived directly from the document text.

        Args:
            file_path: Absolute or relative path to PDF document.
            force_ocr: Bypass local OCR disk cache if True.
            force_pii: Bypass PII cache if True.

        Returns:
            Discriminated union: IngestionSuccess, IngestionEscalated, or IngestionFailed.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Income document not found: {file_path}")

        doc_sha = calculate_sha256(file_path)
        doc_name = os.path.basename(file_path)

        # 1. OCR Extraction
        logger.info("Running OCR extraction on %s using %s...", doc_name, self.ocr_parser.__name__)
        pages = self.ocr_parser.parse_pdf(file_path, force_parsing=force_ocr)
        combined_text = "\n\n".join(p.combined_content for p in pages)

        # 2. PII Sanitization
        logger.info("Sanitizing PII in document text...")
        masked_text, pii_session = self.pii_pipeline.anonymize_text(
            combined_text,
            force_reprocessing=force_pii,
            document_name=doc_name,
        )

        # 3. Multi-Voter Consensus Extraction
        extraction_prompt = (
            "Please extract all Employment Detail Summary (EDS) fields from the following "
            f"official Revenue document text:\n\n"
            f"```text\n{masked_text}\n```"
        )

        logger.info("Executing multi-voter consensus extraction for EDS...")
        consensus_result = run_multi_voter_consensus(
            prompt=extraction_prompt,
            system_prompt=IRISH_EDS_SYSTEM_PROMPT,
            schema_cls=IrishEmploymentDetailSummaryPayload,
            runners=self.runners,
        )

        if consensus_result.status == "failed":
            logger.warning("EDS consensus failed: %s", consensus_result.discrepancies)
            return IngestionFailed(
                error_message=f"Consensus failed: {consensus_result.discrepancies}",
                consensus_result=consensus_result,
            )

        # 4. De-anonymization
        # De-anonymize all voter outputs for staging audit trail
        deanonymized_voters: list[IrishEmploymentDetailSummaryPayload] = []
        for v_out in consensus_result.voter_outputs:
            v_deanon = self.pii_pipeline.deanonymize_item(v_out, pii_session)
            deanonymized_voters.append(v_deanon)

        # De-anonymize reconciled payload if consensus reached
        reconciled_payload: IrishEmploymentDetailSummaryPayload | None = None
        if consensus_result.reconciled_output is not None:
            reconciled_payload = self.pii_pipeline.deanonymize_item(
                consensus_result.reconciled_output,
                pii_session,
            )
            tax_year = reconciled_payload.tax_year
            income_type = reconciled_payload.income_type
        elif deanonymized_voters:
            tax_year = deanonymized_voters[0].tax_year
            income_type = deanonymized_voters[0].income_type
        else:
            return IngestionFailed(
                error_message="No voter outputs available to construct staged record.",
                consensus_result=consensus_result,
            )

        # 5. Determine Verification Status
        if consensus_result.status == "approved":
            if len(consensus_result.discrepancies) == 0:
                verification_status = "auto_approved"
            else:
                verification_status = "majority_agreed"
        else:
            verification_status = "escalated_to_user"

        # 6. Persist to Staging Database Table
        strict_staged = StrictStagedTaxIncomeRecord(
            tax_year=tax_year,
            jurisdiction="ireland",
            income_type=income_type,
            source_document_sha=doc_sha,
            source_file_name=doc_name,
            payload=reconciled_payload,
            voter_outputs=deanonymized_voters,
            discrepancies=consensus_result.discrepancies if consensus_result.discrepancies else None,
            verification_status=verification_status,
            created_at=datetime.now(timezone.utc),
        )

        staged_id = self.db.insert_staged_tax_income_record(strict_staged)
        persisted_staged = strict_staged.model_copy(update={"id": staged_id})

        emp_name = reconciled_payload.employer_name if reconciled_payload else "Unresolved"
        logger.info(
            "Successfully staged tax income record #%d for year %d (%s) with status '%s'.",
            staged_id,
            tax_year,
            emp_name,
            verification_status,
        )


        if consensus_result.status == "escalated":
            return IngestionEscalated(
                staged_record=persisted_staged,
                consensus_result=consensus_result,
            )

        return IngestionSuccess(
            staged_record=persisted_staged,
            consensus_result=consensus_result,
        )
