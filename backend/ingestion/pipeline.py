import hashlib
import json
import os
import re
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, TypeAdapter, model_validator
from sqlmodel import Session

from backend.consensus_models import (
    ConsensusLog,
    MismatchItem,
    TransactionExtractionItem,
)
from backend.db_manager import DatabaseManager
from backend.db_models import (
    FinancialRecord,
    StagedFinancialRecord,
)
from backend.domain_models import IngestionStatus
from backend.ingestion.helpers import IngestionDocument, sanitize_filename
from backend.ingestion.openfigi import OpenFIGIMapper
from backend.ingestion.parser import get_parser_registry
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.ingestion.pii.session import PIISession
from backend.llm.runner import BaseLLMRunner


class VerificationStatus(str, Enum):
    APPROVED = "approved"
    ESCALATED_TO_USER = "escalated_to_user"
    PENDING_VERIFICATION = "pending_verification"
    SKIPPED = "skipped"


class ConsensusVerificationResult(BaseModel):
    status: VerificationStatus
    candidate_records: list[TransactionExtractionItem]
    consensus_log: ConsensusLog

    @model_validator(mode="after")
    def validate_mismatch_indices(self) -> "ConsensusVerificationResult":
        """Ensures mismatches count does not exceed candidate_records and all mismatch indices are in bounds."""
        if self.status == VerificationStatus.APPROVED and self.consensus_log.mismatches:
            raise ValueError("Approved consensus result cannot contain mismatch items.")

        total_candidates = len(self.candidate_records)
        if len(self.consensus_log.mismatches) > total_candidates > 0:
            raise ValueError(
                f"Mismatch count ({len(self.consensus_log.mismatches)}) cannot exceed candidate records count ({total_candidates})."
            )

        for mis in self.consensus_log.mismatches:
            if total_candidates > 0 and not (0 <= mis.index < total_candidates):
                raise ValueError(
                    f"Mismatch index {mis.index} out of bounds for candidate_records of length {total_candidates}."
                )

        return self


class TransactionExtractor:
    """
    Extracts structured transactions from raw text/tables parsed from PDF documents
    using three voter LLMs for judge-style consensus.
    """

    MAX_ATTEMPT_COUNT = 3

    def __init__(self, runners: list[BaseLLMRunner]) -> None:
        """
        Initialize the extractor with exactly three LLM runners representing the voters.

        Args:
            runners: List of three BaseLLMRunner instances.
        """
        if not runners or len(runners) != 3:
            raise ValueError("TransactionExtractor requires exactly three LLM runners (voters).")
        self.runners: list[BaseLLMRunner] = runners

    def _get_system_prompt(self) -> str:
        """
        Dynamically generates the system prompt containing the Pydantic transaction JSON schema.

        Returns:
            A detailed system prompt string with the exact JSON output schema.
        """
        schema_json = json.dumps(TransactionExtractionItem.model_json_schema(), indent=2)

        return (
            "You are a highly precise tax compliance data extractor.\n"
            "Your objective is to extract a chronological list of all financial transactions "
            "and factual tax events from the text, including stock/ETF purchases or sales, "
            "dividends, deposits, tax withholdings, tax payments (like F24), or salaries.\n\n"
            "CRITICAL INSTRUCTIONS FOR KEY FIELDS:\n"
            "1. SYMBOL VS ISIN:\n"
            "   - 'symbol': The short ticker symbol (e.g., AAPL, VUAA, AVGO). Never place the ISIN here.\n"
            "   - 'isin': The 12-character alphanumeric code starting with a country prefix (e.g., US0378331005, IE00BFWXDV39).\n"
            "2. FOREIGN EXCHANGE RATE ('fx_rate'):\n"
            "   - Extract the foreign exchange rate from the transaction currency to the local currency (EUR) if explicitly mentioned in the document.\n"
            "   - If the transaction is already in EUR, or if no exchange rate is mentioned, set 'fx_rate' to null.\n"
            "3. DICT KEYS:\n"
            "   - You MUST output exactly the fields defined in the schema. Do not rename keys (e.g. use event_date, not date; total_amount, not net_amount).\n"
            "4. MISSING VALUES:\n"
            "   - If a field is missing or not applicable, set it to null.\n"
            "5. STANDARDIZATION & FORMATTING:\n"
            "   - 'event_date': Format strictly as ISO 8601 (YYYY-MM-DDTHH:MM:SS) using the exact hour/minute/second from the text.\n"
            "   - 'provider': Standardize name (e.g. use 'DIRECTA SIM', not 'DIRECTA SIM S.P.A.'). Do not include legal entity suffixes like S.p.A., Ltd, Inc.\n"
            "   - 'symbol': Always uppercase.\n"
            "   - 'isin': Always uppercase.\n"
            "   - 'currency': Always uppercase 3-letter code (e.g., EUR).\n"
            "6. ITALIAN NUMBER FORMATTING:\n"
            "   - Italian documents use dot ('.') as a thousands separator ('migliaia') and comma (',') as a decimal point.\n"
            "   - For example, '1.091' means 1091 (one thousand ninety-one shares), NOT 1.091.\n"
            "   - '1.091,50' means 1091.50.\n"
            "   - Always output numbers in standard float/Decimal format using dot ('.') for the decimal point and NO thousands separator.\n"
            "   - Numbers ('quantity', 'unit_price', 'total_amount', 'fx_rate'): Output as standard floats/numbers. 'quantity' and 'total_amount' MUST ALWAYS be positive numbers (e.g., 10.0, never negative -10.0); the 'action' field ('buy', 'sell') specifies direction.\n"
            "6. GROUNDING & SOURCE TRUTH:\n"
            "   - All extracted data must come strictly from the provided document text (this is your absolute source of truth).\n"
            "   - If any data or field cannot be inferred directly from the text, set it to null (even if that field is marked as required by the schema). Do not invent values, do not add conversational notes/comments, just output null.\n"
            "   - Before finalized output, review and check each extracted element to verify it is explicitly present in the source text.\n\n"
            "Format the output strictly as a JSON array of objects matching this JSON schema:\n"
            f"{schema_json}\n\n"
            "Output ONLY the raw JSON array. Do not include markdown code block wrappers (like ```json), "
            "conversational text, or intro explanations."
        )

    def parse_llm_json(self, raw_text: str) -> list[TransactionExtractionItem]:
        """
        Extract and parse a JSON array from model output into Pydantic models.
        Raises ValueError if JSON is missing, malformed, or non-compliant.
        """
        cleaned = BaseLLMRunner.clean_json_response(raw_text)

        # Attempt to extract JSON array from output response if response contains undesired context
        if not (cleaned.startswith("[") or cleaned.startswith("{")):
            # Detect JSON brackets from raw response
            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx : end_idx + 1].strip()
            else:
                raise ValueError("No JSON array block ('[...]') detected in the output.")

        # 2. Parse JSON
        try:
            parsed_raw: object = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Malformed JSON: {e}")

        if not isinstance(parsed_raw, list):
            raise ValueError("Expected a JSON array/list, but parsed object is not a list.")

        parsed_items: list[object] = parsed_raw  # pyright: ignore[reportUnknownVariableType]
        validated_items: list[TransactionExtractionItem] = []
        for idx, item in enumerate(parsed_items):
            try:
                validated_items.append(TransactionExtractionItem.model_validate(item))
            except Exception as e:
                raise ValueError(f"Non-compliant JSON at index {idx}: {e}")

        return validated_items

    def _save_voter_log_attempt(
        self,
        log_dir: str,
        document_name: str,
        voter_index: int,
        attempt: int,
        system_prompt: str,
        prompt: str,
        raw_response: str,
    ) -> None:
        """Save raw voter response to a log file."""
        try:
            os.makedirs(log_dir, exist_ok=True)
            safe_doc_name = sanitize_filename(document_name)
            log_filename = os.path.join(log_dir, f"{safe_doc_name}_voter_{voter_index}_attempt_{attempt}.txt")
            with open(log_filename, "w", encoding="utf-8") as f:
                _ = f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
                _ = f.write(f"=== PROMPT ===\n{prompt}\n\n")
                _ = f.write(f"=== RAW RESPONSE ===\n{raw_response}\n")
            print(f"    - Saved raw voter #{voter_index} log: {log_filename}")
        except Exception as log_err:
            print(f"    - ⚠️ Warning: Failed to save raw response to logs: {log_err}")

    def _query_voter_with_retry(
        self,
        voter_index: int,
        runner: BaseLLMRunner,
        prompt: str,
        system_prompt: str,
        document_name: str,
        run_id: str | None,
        save_logs: bool,
    ) -> list[TransactionExtractionItem]:
        """Query LLM runner with retries and validate response schema."""
        parsed_list = None
        last_error = None
        max_attempts = self.MAX_ATTEMPT_COUNT
        log_dir = os.path.join("logs", run_id) if run_id else "logs"

        for attempt in range(1, max_attempts + 1):
            model_name = runner.model_name
            print(
                f"  - **Querying Voter #{voter_index}** (model: \033[1m{model_name}\033[0m) - Attempt {attempt}/{max_attempts}..."
            )
            raw_response = runner.complete(prompt, system_prompt)

            if save_logs:
                self._save_voter_log_attempt(
                    log_dir, document_name, voter_index, attempt, system_prompt, prompt, raw_response
                )

            try:
                parsed_list = self.parse_llm_json(raw_response)
                break  # Valid JSON and validation passed!
            except Exception as e:
                last_error = str(e)
                print(f"    ⚠️ Voter #{voter_index} parse/validation failed on attempt {attempt}: {last_error}")

        if parsed_list is None:
            raise ValueError(
                f"Voter #{voter_index} failed to return a compliant JSON schema after {max_attempts} attempts.\n"
                + f"Last Extraction Error: {last_error}"
            )

        return parsed_list

    def _load_cached_voter_extractions(
        self,
        file_sha: str,
        combined_text: str,
        system_prompt: str,
        models_key: str,
        force_reprocessing: bool,
    ) -> tuple[list[list[TransactionExtractionItem]] | None, str]:
        """Attempt to load voter extractions from disk cache if signature matches."""
        content_sig = f"{file_sha}_{combined_text}_{system_prompt}_{models_key}"
        voter_hash = hashlib.sha256(content_sig.encode("utf-8")).hexdigest()
        cache_dir = ".cache/voters"
        cache_path = os.path.join(cache_dir, f"{voter_hash}_voters.json")

        if not force_reprocessing and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached_raw = json.load(f)
                adapter = TypeAdapter(list[list[TransactionExtractionItem]])
                voter_extractions = adapter.validate_python(cached_raw)
                print("  - **Loaded cached voter extractions** (content & prompt signature verified).")
                return voter_extractions, cache_path
            except Exception as cache_err:
                print(f"  - ⚠️ Warning: Failed to load voter cache '{cache_path}': {cache_err}")

        return None, cache_path

    def _save_cached_voter_extractions(
        self,
        cache_path: str,
        voter_extractions: list[list[TransactionExtractionItem]],
    ) -> None:
        """Save voter extractions to disk cache."""
        try:
            cache_dir = ".cache/voters"
            os.makedirs(cache_dir, exist_ok=True)
            serializable_data = [
                [item.model_dump(mode="json") for item in voter_list] for voter_list in voter_extractions
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, default=str)
        except Exception as cache_err:
            print(f"  - ⚠️ Warning: Failed to save voter response cache: {cache_err}")

    def extract_voter_extractions(
        self,
        document_name: str,
        combined_text: str,
        jurisdiction: str,
        run_id: str | None = None,
        save_logs: bool = True,
        file_sha: str | None = None,
        force_reprocessing: bool = False,
    ) -> list[list[TransactionExtractionItem]]:
        """
        Query the three voter roles separately and return their respective extracted transaction lists.
        Caches voter output to disk by file_sha to prevent redundant LLM voter queries across runs.

        Args:
            document_name: Name of the raw PDF file.
            combined_text: Extracted and concatenated layout text content.
            jurisdiction: Jurisdiction code ('italy' or 'ireland').
            run_id: Optional timestamped run folder name for logging.
            save_logs: If True, save voter responses to the logs directory.
            file_sha: SHA-256 hash of the input file for voter response disk caching.
            force_reprocessing: If True, bypass voter response cache and re-query LLMs.

        Returns:
            A list containing three sub-lists of TransactionExtractionItem (one list per voter).
        """
        system_prompt = self._get_system_prompt()
        models_key = "_".join(r.model_name for r in self.runners)

        cache_path: str | None = None
        if file_sha:
            cached_extractions, cache_path = self._load_cached_voter_extractions(
                file_sha, combined_text, system_prompt, models_key, force_reprocessing
            )
            if cached_extractions is not None:
                return cached_extractions

        voter_extractions: list[list[TransactionExtractionItem]] = []

        for idx, runner in enumerate(self.runners, start=1):
            prompt = (
                f"Document Name: {document_name}\n"
                f"Jurisdiction Context: {jurisdiction}\n"
                f"Document Text Content:\n{combined_text}\n\n"
                f"You are Voter Agent #{idx}."
            )
            parsed_list = self._query_voter_with_retry(
                idx, runner, prompt, system_prompt, document_name, run_id, save_logs
            )
            voter_extractions.append(parsed_list)

        if cache_path:
            self._save_cached_voter_extractions(cache_path, voter_extractions)

        return voter_extractions


class ConsensusVerifier:
    """Compare Voter extractions and verify 100% field consensus."""

    @classmethod
    def records_are_equivalent(cls, r1: TransactionExtractionItem, r2: TransactionExtractionItem) -> bool:
        """Compare all key fields after normalization to determine equivalence."""
        # 1. Compare datetime objects at the date and hour level
        d1 = r1.event_date
        d2 = r2.event_date
        if (d1.year, d1.month, d1.day, d1.hour) != (d2.year, d2.month, d2.day, d2.hour):
            return False

        # 2. Compare Enums directly
        if r1.asset_type != r2.asset_type:
            return False
        if r1.action != r2.action:
            return False

        # 3. Compare currency directly (case-insensitive)
        if r1.currency.strip().upper() != r2.currency.strip().upper():
            return False

        # 4. Compare Decimal fields (with tolerance)
        def decimal_equivalent(v1: Decimal | None, v2: Decimal | None) -> bool:
            if v1 is None and v2 is None:
                return True
            if v1 is None or v2 is None:
                return False
            return abs(v1 - v2) < Decimal("0.000001")

        if not decimal_equivalent(r1.quantity, r2.quantity):
            return False
        if not decimal_equivalent(r1.unit_price, r2.unit_price):
            return False
        if not decimal_equivalent(r1.fees, r2.fees):
            return False
        if not decimal_equivalent(r1.total_amount, r2.total_amount):
            return False
        if not decimal_equivalent(r1.fx_rate, r2.fx_rate):
            return False

        # 5. Compare identifier/provider strings with alphanumeric normalization
        def clean_str(s: str | None) -> str:
            if not s:
                return ""
            val = s.strip().upper()
            # Strip common legal suffixes
            val = re.sub(r"\b(S\.?P\.?A\.?|LTD|INC|CORP|CO|LIMITED)\b", "", val)
            # Remove all non-alphanumeric characters
            return re.sub(r"[^A-Z0-9]", "", val)

        if clean_str(r1.symbol) != clean_str(r2.symbol):
            return False
        if clean_str(r1.isin) != clean_str(r2.isin):
            return False
        if clean_str(r1.provider) != clean_str(r2.provider):
            return False

        return True

    @classmethod
    def calculate_similarity_score(cls, r1: TransactionExtractionItem, r2: TransactionExtractionItem) -> float:
        """
        Calculate weighted similarity score between two records (0.0 to 1.0 / 0% to 100%).
        """
        score = 0.0

        # 1. Date similarity (max 0.25)
        d1, d2 = r1.event_date, r2.event_date
        if (d1.year, d1.month, d1.day, d1.hour) == (d2.year, d2.month, d2.day, d2.hour):
            score += 0.25
        elif (d1.year, d1.month, d1.day) == (d2.year, d2.month, d2.day):
            score += 0.20
        elif abs((d1 - d2).total_seconds()) <= 172800:
            score += 0.03

        # 2. Action & Asset Type (max 0.20)
        if r1.action == r2.action:
            score += 0.10
        if r1.asset_type == r2.asset_type:
            score += 0.10

        # 3. Symbol / ISIN (max 0.25)
        s1 = (r1.symbol or "").strip().upper()
        s2 = (r2.symbol or "").strip().upper()
        isin1 = (r1.isin or "").strip().upper()
        isin2 = (r2.isin or "").strip().upper()

        if (s1 and s1 == s2) or (isin1 and isin1 == isin2):
            score += 0.25
        elif s1 and s2 and (s1 in s2 or s2 in s1):
            score += 0.15

        # 4. Total Amount & Quantity (max 0.30)
        if r1.total_amount is not None and r2.total_amount is not None:
            if abs(r1.total_amount - r2.total_amount) < Decimal("0.01"):
                score += 0.15
            elif r1.total_amount > 0 and abs(r1.total_amount - r2.total_amount) / r1.total_amount <= Decimal("0.05"):
                score += 0.10

        if r1.quantity is not None and r2.quantity is not None:
            if abs(r1.quantity - r2.quantity) < Decimal("0.0001"):
                score += 0.15
            elif r1.quantity > 0 and abs(r1.quantity - r2.quantity) / r1.quantity <= Decimal("0.05"):
                score += 0.10

        return round(score, 2)

    @classmethod
    def verify_consensus(cls, votes: list[list[TransactionExtractionItem]]) -> ConsensusVerificationResult:
        """
        Verify consensus between three lists of voter extractions.

        Args:
            votes: List containing three lists of TransactionExtractionItem.

        Returns:
            A ConsensusVerificationResult object detailing status, merged records, and consensus log.
        """
        if len(votes) != 3:
            raise ValueError(f"Consensus verification requires exactly 3 voter lists, got {len(votes)}")

        v1, v2, v3 = votes

        def sort_key(r: TransactionExtractionItem) -> tuple[str, str, str]:
            dt_str = r.event_date.isoformat()
            return (dt_str, r.symbol or "", r.action.value)

        v1_sorted = sorted(v1, key=sort_key)
        v2_sorted = sorted(v2, key=sort_key)
        v3_sorted = sorted(v3, key=sort_key)

        # 1. Compare list lengths
        if len(v1) != len(v2) or len(v2) != len(v3):
            candidates: list[TransactionExtractionItem] = []
            for voter_list in (v1_sorted, v2_sorted, v3_sorted):
                for item in voter_list:
                    if not any(cls.records_are_equivalent(item, c) for c in candidates):
                        candidates.append(item)
            candidates.sort(key=sort_key)

            count_mismatches: list[MismatchItem] = []
            for idx, cand in enumerate(candidates):

                def best_match(
                    voter_list: list[TransactionExtractionItem],
                ) -> tuple[TransactionExtractionItem | None, float]:
                    best_item = None
                    best_score = 0.0
                    for item in voter_list:
                        score = cls.calculate_similarity_score(item, cand)
                        if score > best_score:
                            best_score = score
                            best_item = item
                    if best_score >= 0.40:
                        return best_item, best_score
                    return None, 0.0

                m1, s1 = best_match(v1_sorted)
                m2, s2 = best_match(v2_sorted)
                m3, s3 = best_match(v3_sorted)

                valid_scores = [s for s in (s1, s2, s3) if s > 0]
                avg_sim = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0

                present_count = sum(1 for m in (m1, m2, m3) if m is not None)
                if present_count >= 2:
                    count_mismatches.append(
                        MismatchItem(
                            index=idx,
                            voter1=m1,
                            voter2=m2,
                            voter3=m3,
                            similarity_score=avg_sim,
                        )
                    )

            return ConsensusVerificationResult(
                status=VerificationStatus.ESCALATED_TO_USER,
                candidate_records=candidates,
                consensus_log=ConsensusLog(
                    version="1.0",
                    error="Mismatch in transaction counts",
                    mismatches=count_mismatches,
                    raw_voter_1_records=v1_sorted,
                    raw_voter_2_records=v2_sorted,
                    raw_voter_3_records=v3_sorted,
                ),
            )

        # 2. Equal length: Check field-by-field consensus per index
        has_consensus = True
        mismatches: list[MismatchItem] = []

        for idx in range(len(v1_sorted)):
            r1 = v1_sorted[idx]
            r2 = v2_sorted[idx]
            r3 = v3_sorted[idx]

            if not cls.records_are_equivalent(r1, r2) or not cls.records_are_equivalent(r2, r3):
                has_consensus = False
                s12 = cls.calculate_similarity_score(r1, r2)
                s23 = cls.calculate_similarity_score(r2, r3)
                s13 = cls.calculate_similarity_score(r1, r3)
                avg_similarity = round((s12 + s23 + s13) / 3.0, 2)

                mismatches.append(
                    MismatchItem(
                        index=idx,
                        voter1=r1,
                        voter2=r2,
                        voter3=r3,
                        similarity_score=avg_similarity,
                    )
                )

        if has_consensus:
            return ConsensusVerificationResult(
                status=VerificationStatus.APPROVED,
                candidate_records=v1_sorted,
                consensus_log=ConsensusLog(
                    version="1.0",
                    message="Unanimous consensus reached.",
                    raw_voter_1_records=v1_sorted,
                    raw_voter_2_records=v2_sorted,
                    raw_voter_3_records=v3_sorted,
                ),
            )

        candidates = []
        for voter_list in (v1_sorted, v2_sorted, v3_sorted):
            for item in voter_list:
                if not any(cls.records_are_equivalent(item, c) for c in candidates):
                    candidates.append(item)
        candidates.sort(key=sort_key)

        return ConsensusVerificationResult(
            status=VerificationStatus.ESCALATED_TO_USER,
            candidate_records=candidates,
            consensus_log=ConsensusLog(
                version="1.0",
                error="Field value mismatch",
                mismatches=mismatches,
                raw_voter_1_records=v1_sorted,
                raw_voter_2_records=v2_sorted,
                raw_voter_3_records=v3_sorted,
            ),
        )


class TransactionPipeline:
    """Orchestrates PDF parsing, voter consensus extraction, and database persistence."""

    def __init__(
        self, db: DatabaseManager, runners: list[BaseLLMRunner], pii_pipeline: PIIPipeline, save_logs: bool = True
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            db: DatabaseManager instance.
            runners: List of three BaseLLMRunner instances.
            save_logs: Whether to log raw completions to local disk.
            pii_pipeline: pre-configured PIIPipeline instance.
        """
        self.db: DatabaseManager = db
        self.extractor: TransactionExtractor = TransactionExtractor(runners)
        self.save_logs: bool = save_logs
        self.run_id: str = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_log_dir: str = os.path.join("logs", self.run_id)
        self.figi_mapper: OpenFIGIMapper = OpenFIGIMapper()
        self.pii_pipeline: PIIPipeline = pii_pipeline

    def _save_pii_audit_logs(
        self, document_name: str, combined_text: str, masked_text: str, pii_session: PIISession
    ) -> None:
        """Save PII audit logs for verification."""
        try:
            log_dir = os.path.join("logs", self.run_id)
            os.makedirs(log_dir, exist_ok=True)
            safe_doc_name = sanitize_filename(document_name)

            with open(os.path.join(log_dir, f"{safe_doc_name}_raw_unmasked.txt"), "w", encoding="utf-8") as f:
                _ = f.write(combined_text)
            with open(os.path.join(log_dir, f"{safe_doc_name}_raw_masked.txt"), "w", encoding="utf-8") as f:
                _ = f.write(masked_text)
            with open(os.path.join(log_dir, f"{safe_doc_name}_pii_mapping.json"), "w", encoding="utf-8") as f:
                json.dump(pii_session.placeholder_map, f, indent=2)
            print(f"  - **Saved PII audit logs to**: {log_dir}")
        except Exception as log_err:
            print(f"  - ⚠️ **Warning**: Failed to save PII audit logs: {log_err}")

    def _resolve_local_amount(
        self, currency: str, total_amount: Decimal, fx_rate: Decimal | None
    ) -> tuple[Decimal | None, bool]:
        """
        Resolve local_total_amount and check if escalation is needed due to missing FX rate.
        Returns (local_total_amount, needs_escalation).
        """
        if currency == "EUR":
            return total_amount, False
        if fx_rate is not None:
            return total_amount * fx_rate, False
        return None, True

    def _parse_and_anonymize_document(
        self,
        doc: IngestionDocument,
        parser: str,
        force_ocr: bool,
        force_pii_reprocessing: bool,
    ) -> tuple[str, PIISession]:
        """Parse PDF document text and apply PII anonymization.

        Args:
            doc: Target IngestionDocument entity.
            parser: Name of registered parser interface to invoke.
            force_ocr: If True, force re-running OCR parser/API from scratch.
            force_pii_reprocessing: If True, bypass cached PII anonymization results.

        Returns:
            Tuple of (masked_text_content, active_pii_session).

        Raises:
            ValueError: If the parser name is not found in the parser registry.
        """
        parser_registry = get_parser_registry()
        if parser not in parser_registry:
            raise ValueError(f"Unknown parser '{parser}'. Available options: {list(parser_registry.keys())}")
        parser_class = parser_registry[parser]
        pages = parser_class.parse_pdf(doc.file_path, force_parsing=force_ocr)
        combined_text = "\n\n".join(p.combined_content for p in pages)

        masked_text, pii_session = self.pii_pipeline.anonymize_text(
            combined_text,
            force_reprocessing=force_pii_reprocessing,
            document_name=doc.doc_name,
            log_dir=self.run_log_dir if self.save_logs else None,
        )

        if self.save_logs:
            self._save_pii_audit_logs(doc.doc_name, combined_text, masked_text, pii_session)

        return masked_text, pii_session

    def _deanonymize_consensus_results(
        self,
        candidate_records: list[TransactionExtractionItem],
        consensus_log: ConsensusLog,
        pii_session: PIISession,
    ) -> tuple[list[TransactionExtractionItem], ConsensusLog]:
        """De-anonymize candidate items and voter extractions in consensus log.

        Args:
            candidate_records: Extracted candidate items to de-anonymize.
            consensus_log: Consensus verification log container.
            pii_session: Active PII mapping session.

        Returns:
            Tuple of (deanonymized_candidate_records, deanonymized_consensus_log).
        """
        deanon_matched_items: list[TransactionExtractionItem] = [
            self.pii_pipeline.deanonymize_item(item, pii_session) for item in candidate_records
        ]

        if consensus_log.mismatches:
            for mis in consensus_log.mismatches:
                if mis.voter1:
                    mis.voter1 = self.pii_pipeline.deanonymize_item(mis.voter1, pii_session)
                if mis.voter2:
                    mis.voter2 = self.pii_pipeline.deanonymize_item(mis.voter2, pii_session)
                if mis.voter3:
                    mis.voter3 = self.pii_pipeline.deanonymize_item(mis.voter3, pii_session)

        if consensus_log.raw_voter_1_records:
            consensus_log.raw_voter_1_records = [
                self.pii_pipeline.deanonymize_item(r, pii_session) for r in consensus_log.raw_voter_1_records
            ]
        if consensus_log.raw_voter_2_records:
            consensus_log.raw_voter_2_records = [
                self.pii_pipeline.deanonymize_item(r, pii_session) for r in consensus_log.raw_voter_2_records
            ]
        if consensus_log.raw_voter_3_records:
            consensus_log.raw_voter_3_records = [
                self.pii_pipeline.deanonymize_item(r, pii_session) for r in consensus_log.raw_voter_3_records
            ]

        return deanon_matched_items, consensus_log

    def _persist_staged_records(
        self,
        doc: IngestionDocument,
        candidate_items: list[TransactionExtractionItem],
        consensus_status: VerificationStatus,
        consensus_log: ConsensusLog,
        force: bool,
    ) -> tuple[VerificationStatus, list[StagedFinancialRecord]]:
        """Construct StagedFinancialRecord entities and write to database atomically.

        Args:
            doc: Target IngestionDocument entity.
            candidate_items: De-anonymized candidate transaction items.
            consensus_status: Consensus verification status from ConsensusVerifier.
            consensus_log: Detailed consensus verification audit log.
            force: If True, delete pre-existing staged and financial records for this document.

        Returns:
            Tuple of (final_verification_status, persisted_staged_records).
        """
        records: list[StagedFinancialRecord] = []
        records_to_insert: list[StagedFinancialRecord] = []
        final_status = consensus_status

        for item in candidate_items:
            resolved_symbol = item.symbol
            resolved_asset_name = item.asset_name
            openfigi_detected_symbol = None
            if item.isin:
                figi_res = self.figi_mapper.map_isin(item.isin)
                figi_symbol, figi_name = figi_res.ticker, figi_res.name
                if figi_symbol:
                    openfigi_detected_symbol = figi_symbol
                    if not resolved_symbol:
                        resolved_symbol = figi_symbol
                if figi_name and not resolved_asset_name:
                    resolved_asset_name = figi_name

            quantity = item.quantity
            unit_price = item.unit_price
            fees = item.fees
            total_amount = item.total_amount

            local_total_amount, needs_escalation = self._resolve_local_amount(item.currency, total_amount, item.fx_rate)

            item_status_str = (
                "escalated_to_user"
                if (consensus_status == VerificationStatus.ESCALATED_TO_USER or needs_escalation)
                else "pending_approval"
            )

            if needs_escalation:
                final_status = VerificationStatus.ESCALATED_TO_USER

            tax_year = item.event_date.year

            record = StagedFinancialRecord(
                provider=doc.provider,
                source_file_name=doc.doc_name,
                source_file_sha=doc.sha,
                event_timestamp=item.event_date,
                asset_type=item.asset_type.value,
                symbol=resolved_symbol,
                isin=item.isin,
                asset_name=resolved_asset_name,
                action=item.action.value,
                quantity=quantity,
                unit_price=unit_price,
                currency=item.currency,
                fees=fees,
                total_amount=total_amount,
                fx_rate=item.fx_rate,
                local_total_amount=local_total_amount,
                tax_year=tax_year,
                account_country=doc.account_country,
                verification_status=item_status_str,
                consensus_log=consensus_log.model_dump_json(),
                openfigi_detected=openfigi_detected_symbol,
            )
            records_to_insert.append(record)

        with Session(self.db.engine) as session:
            if force:
                existing_staged = self.db.get_staged_records(account_country=doc.account_country, source_file_sha=doc.sha)
                existing_financial = self.db.get_financial_records(
                    account_country=doc.account_country, source_file_sha=doc.sha
                )
                for r in existing_staged:
                    if r.id is not None:
                        db_rec = session.get(StagedFinancialRecord, r.id)
                        if db_rec:
                            session.delete(db_rec)
                for r in existing_financial:
                    if r.id is not None:
                        db_rec = session.get(FinancialRecord, r.id)
                        if db_rec:
                            session.delete(db_rec)

            for record in records_to_insert:
                session.add(record)

            session.commit()

            for record in records_to_insert:
                session.refresh(record)
                records.append(record)

        self.db.upsert_ingested_source_document(
            file_sha=doc.sha,
            file_name=doc.doc_name,
            provider=doc.provider or "",
            account_country=doc.account_country or "",
            status=IngestionStatus.SUCCESS,
            transaction_count=len(records),
        )

        print(f"  - **Staged {len(records)} transaction(s)**. Status: \033[1m{final_status.name}\033[0m")
        return final_status, records

    def ingest_records_document(
        self,
        doc: IngestionDocument,
        parser: str,
        force: bool = False,
        force_pii_reprocessing: bool = False,
        force_ocr: bool = False,
    ) -> tuple[VerificationStatus, list[FinancialRecord] | list[StagedFinancialRecord]]:
        """Parse PDF document, extract transaction records with consensus, and persist to database.

        Args:
            doc: IngestionDocument model containing path, name, sha, provider, and jurisdiction.
            parser: PDF extraction backend parser to use ('pdfplumber', 'chandra', or 'chandra_api').
            force: If True, overwrite existing database entries for this document.
            force_pii_reprocessing: If True, force re-processing of PII anonymization bypass cache.
            force_ocr: If True, bypass local OCR cache and force raw parsing/OCR from scratch.

        Returns:
            A tuple of (VerificationStatus, persisted_records).

        Raises:
            ValueError: If mandatory jurisdiction is missing or parser is unknown.
        """
        if self.db.is_source_document_ingested(doc.sha) and not force:
            existing_staged = self.db.get_staged_records(account_country=doc.account_country, source_file_sha=doc.sha)
            existing_financial = self.db.get_financial_records(account_country=doc.account_country, source_file_sha=doc.sha)
            print(f"  - **Skipping document** '\033[1m{doc.doc_name}\033[0m' (already successfully ingested).")
            return VerificationStatus.SKIPPED, existing_staged or existing_financial

        try:
            if force:
                print(f"  - **Re-ingesting document** '\033[1m{doc.doc_name}\033[0m' (--force active).")

            masked_text, pii_session = self._parse_and_anonymize_document(
                doc, parser, force_ocr, force_pii_reprocessing
            )

            if not doc.account_country:
                raise ValueError(f"IngestionDocument '{doc.doc_name}' is missing mandatory 'account_country' attribute.")

            print(f"  - **Run Judge-Style consensus voting on**: \033[1m{doc.doc_name}\033[0m...")
            votes = self.extractor.extract_voter_extractions(
                doc.doc_name,
                masked_text,
                doc.account_country,
                run_id=self.run_id,
                save_logs=self.save_logs,
                file_sha=doc.sha,
                force_reprocessing=force,
            )

            result = ConsensusVerifier.verify_consensus(votes)
            print(f"  - **Verification Status**: \033[1m{result.status.name}\033[0m")

            deanon_items, deanon_log = self._deanonymize_consensus_results(
                result.candidate_records, result.consensus_log, pii_session
            )

            return self._persist_staged_records(doc, deanon_items, result.status, deanon_log, force)

        except Exception as e:
            self.db.upsert_ingested_source_document(
                file_sha=doc.sha,
                file_name=doc.doc_name,
                provider=doc.provider or "",
                account_country=doc.account_country or "",
                status=IngestionStatus.FAILED,
                transaction_count=0,
            )
            raise e
