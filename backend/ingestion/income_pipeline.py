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

from backend.db_manager import DatabaseManager
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictTaxIncomeRecord,
)
from backend.ingestion.generic_voter import GenericConsensusResult, run_multi_voter_consensus
from backend.ingestion.helpers import calculate_sha256
from backend.ingestion.ocr_parser import ChandraAPIParser, ChandraOCRParser
from backend.ingestion.parser import BasePDFParser
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.llm.pydantic_ai_runner import PydanticAIRunner

logger = logging.getLogger(__name__)

IRISH_EDS_SYSTEM_PROMPT = """You are an expert Irish tax accounting extraction engine.
Your task is to extract official employment income details from an Irish Revenue "Employment Detail Summary" (EDS / P60 replacement) document into a single JSON object.

Extract the following fields strictly into the JSON schema:
- income_type: "irish_employment_detail_summary"
- employer_name: Full registered name of employer
- employer_registration_number: Employer tax registration number / ERN (or null)
- employment_id: Employment sequence number/ID (or null)
- start_date: ISO 8601 start date (YYYY-MM-DD) if available, else null
- end_date: ISO 8601 end date (YYYY-MM-DD) if available, else null
- gross_pay_eur: Total gross pay for the tax year in EUR (positive Decimal)
- income_tax_paid_eur: Total Income Tax (PAYE) deducted in EUR (positive Decimal)
- usc_paid_eur: Total Universal Social Charge (USC) deducted in EUR (positive Decimal)
- prsi_paid_eur: Total Employee PRSI deducted in EUR (positive Decimal)
- employer_prsi_paid_eur: Total Employer PRSI paid in EUR (positive Decimal, or null)
- prsi_class: PRSI class letter (e.g. "A1", "A", or null)
- prsi_weeks: Number of insurable weeks under PRSI as integer (or null)
- lpt_deducted_eur: Local Property Tax deducted at source in EUR (positive Decimal, or null)

RULES:
1. Strip currency symbols (€, EUR) and format numbers with a decimal dot (e.g. 75000.50).
2. Do not fabricate values. Use null for missing optional fields.
3. Return strictly valid JSON without markdown conversational text.
"""


def _get_default_ocr_parser() -> type[BasePDFParser]:
    """Resolve default OCR parser: ChandraAPIParser if DATALAB_API_KEY set, else ChandraOCRParser."""
    if os.environ.get("DATALAB_API_KEY"):
        return ChandraAPIParser
    return ChandraOCRParser


class IncomeIngestionPipeline:
    """Manages end-to-end ingestion and consensus verification of tax income documents."""

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
        # TODO: can tax year be derived from the document ?
        tax_year: int,
        force_ocr: bool = False,
        force_pii: bool = False,
        # TODO: This could be modeled using a discriminated union of payloads per document type
    ) -> tuple[StrictTaxIncomeRecord | None, GenericConsensusResult[IrishEmploymentDetailSummaryPayload]]:
        """Ingest and verify an Irish Revenue Employment Detail Summary (EDS) PDF.

        Args:
            file_path: Absolute or relative path to PDF document.
            tax_year: Tax year of the employment summary.
            force_ocr: Bypass local OCR disk cache if True.
            force_pii: Bypass PII cache if True.

        Returns:
            Tuple of (persisted StrictTaxIncomeRecord or None if escalated, consensus result).
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
            f"Please extract all Employment Detail Summary (EDS) fields from the following "
            f"official Revenue document text for Irish tax year {tax_year}:\n\n"
            f"```text\n{masked_text}\n```"
        )

        logger.info("Executing multi-voter consensus extraction for EDS...")
        consensus_result = run_multi_voter_consensus(
            prompt=extraction_prompt,
            system_prompt=IRISH_EDS_SYSTEM_PROMPT,
            schema_cls=IrishEmploymentDetailSummaryPayload,
            runners=self.runners,
        )

        if consensus_result.status != "approved" or consensus_result.reconciled_output is None:
            logger.warning(
                "EDS consensus was not approved (status=%s): %s",
                consensus_result.status,
                consensus_result.discrepancies,
            )
            return None, consensus_result

        # 4. De-anonymization
        reconciled_payload = self.pii_pipeline.deanonymize_item(
            consensus_result.reconciled_output,
            pii_session,
        )

        # 5. Persist to Database
        strict_record = StrictTaxIncomeRecord(
            tax_year=tax_year,
            jurisdiction="ireland",
            income_type=reconciled_payload.income_type,
            source_document_sha=doc_sha,
            payload=reconciled_payload,
            created_at=datetime.now(timezone.utc),
        )

        record_id = self.db.insert_tax_income_record(strict_record)
        persisted_record = strict_record.model_copy(update={"id": record_id})
        logger.info(
            "Successfully persisted tax income record #%d for year %d (%s).",
            record_id,
            tax_year,
            reconciled_payload.employer_name,
        )

        return persisted_record, consensus_result
