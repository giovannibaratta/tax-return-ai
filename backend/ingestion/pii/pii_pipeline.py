import hashlib
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, TypeVar, cast

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer, SpacyRecognizer
from pydantic import BaseModel
from returns.result import Failure, Result, Success

from backend.ingestion.pii.constants import (
    DATE_TIME_REGEX,
    EXCLUDE_FROM_PII_SYMBOLS,
    PLACEHOLDER_REGEX,
    READ_CACHE_ERROR_FRIENDLY_MESSAGES,
)
from backend.ingestion.pii.models import AnonymizationPassResult, PIICacheData, PIIPipelineConfig, ReadCacheError
from backend.ingestion.pii.session import PIISession
from backend.llm.runner import BaseLLMRunner

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline

T = TypeVar("T")
TModel = TypeVar("TModel", bound=BaseModel)


class PIIPipeline:
    """
    Independent, stateless PII Anonymization and De-anonymization pipeline.

    Implements three sequential filter passes before text is sent to voter LLMs:

    Pass 1 — LLM-based Redaction (generative, local/remote):
        A first sweep using a trusted generative model to detect and redact
        all names, addresses, and account numbers into decatable placeholders.

    Pass 2 — Presidio (NER-based, local):
        Detects PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, ACCOUNT_NUMBER plus
        custom regex recognizers for Italian Fiscal Code, Irish PPSN, and bank account numbers.
        ORGANIZATION is intentionally excluded to avoid masking broker and asset names.

    Pass 3 — OpenAI Privacy Filter (transformer token-classifier, local):
        A third sweep for any PII that slipped past. Detects:
        private_person, private_address, private_email, private_phone,
        private_url, private_date, account_number, secret.
        Model loaded locally via HuggingFace Transformers
        (openai/privacy-filter or a path set via OPENAI_PRIVACY_FILTER_MODEL_PATH).

    All passes are independently togglable via configuration.
    """

    def __init__(self, config: PIIPipelineConfig | None = None) -> None:
        """Initialize the PII Pipeline.

        Args:
            config: Optional configuration instance. If None, initialized from environment.
        """
        self.config: PIIPipelineConfig = config or PIIPipelineConfig.from_env()

        if not (self.config.presidio_enabled or self.config.openai_filter_enabled or self.config.llm_redaction):
            raise ValueError(
                "All anonymization passes are disabled. " + "At least one pass must be enabled to prevent PII leakage."
            )

        # ── Caching Setup ────────────────────────────────────────────────────
        if self.config.pii_cache_enabled:
            try:
                os.makedirs(self.config.pii_cache_dir, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize PII cache directory '{self.config.pii_cache_dir}': {e}"
                ) from e

        # ── Pass-2: Presidio Setup ───────────────────────────────────────────
        # Configure the NLP engine to support both English (default) and Italian models.
        # This resolves multilingual NER failures (like Italian Person/Location recognition).
        provider_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"},
                {"lang_code": "it", "model_name": "it_core_news_sm"},
            ],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=provider_config).create_engine()
        self.analyzer: AnalyzerEngine = AnalyzerEngine(nlp_engine=nlp_engine)
        self._add_custom_recognizers()

        # ── Pass-3: OpenAI Privacy Filter Setup ─────────────────────────────
        self._target_entities: list[str] = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "ITALIAN_FISCAL_CODE",
            "IRISH_PPSN",
            "LOCATION",
            "ACCOUNT_NUMBER",
        ]
        self._privacy_filter_pipeline: Pipeline | None = None

    # ── Initialization helpers ────────────────────────────────────────────────

    def _add_custom_recognizers(self) -> None:
        """Register custom regex-based recognizers for region-specific tax IDs and accounts."""
        # Presidio automatically registers the built-in Spacy, Email, and Phone recognizers
        # for English ('en'), but NOT for Italian ('it'). We explicitly instantiate and add
        # them with supported_language="it" to support Italian document processing.
        self.analyzer.registry.add_recognizer(SpacyRecognizer(supported_language="it"))
        self.analyzer.registry.add_recognizer(EmailRecognizer(supported_language="it"))
        self.analyzer.registry.add_recognizer(PhoneRecognizer(supported_language="it"))

        for lang in ["en", "it"]:
            # Italian Codice Fiscale
            cf_recognizer = PatternRecognizer(
                supported_entity="ITALIAN_FISCAL_CODE",
                supported_language=lang,
                patterns=[
                    Pattern(
                        name="cf_pattern",
                        regex=r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
                        score=0.85,
                    )
                ],
            )
            self.analyzer.registry.add_recognizer(cf_recognizer)

            # Irish Personal Public Service Number (PPSN)
            ppsn_recognizer = PatternRecognizer(
                supported_entity="IRISH_PPSN",
                supported_language=lang,
                patterns=[
                    Pattern(
                        name="ppsn_pattern",
                        regex=r"\b\d{7}[A-Z]{1,2}\b",
                        score=0.85,
                    )
                ],
            )
            self.analyzer.registry.add_recognizer(ppsn_recognizer)

            # Italian Address/Location Recognizer
            address_recognizer = PatternRecognizer(
                supported_entity="LOCATION",
                supported_language=lang,
                patterns=[
                    Pattern(
                        name="italian_address",
                        regex=r"(?i)\b(?:via|viale|piazza|corso|strada|vicolo|p\.zza|c\.so|v\.le)\s+[A-Za-z\s']+\d+\b",
                        score=0.95,
                    )
                ],
            )
            self.analyzer.registry.add_recognizer(address_recognizer)

            # Italian Postal Code (CAP) + City Recognizer
            cap_recognizer = PatternRecognizer(
                supported_entity="LOCATION",
                supported_language=lang,
                patterns=[
                    Pattern(
                        name="italian_cap",
                        regex=r"(?i)\b\d{5}\s+[A-Za-z\s']+\b",
                        score=0.90,
                    )
                ],
            )
            self.analyzer.registry.add_recognizer(cap_recognizer)

            # Account Number Recognizer
            account_recognizer = PatternRecognizer(
                supported_entity="ACCOUNT_NUMBER",
                supported_language=lang,
                patterns=[
                    Pattern(
                        name="account_number_conto",
                        regex=r"(?i)\b(?:conto|account)(?:\s+number)?\s+([A-Z0-9-]{3,20})\b",
                        score=0.90,
                    ),
                    Pattern(
                        name="account_number_ibkr",
                        regex=r"\bU\*\*\*[0-9]{3,10}\b",
                        score=0.95,
                    ),
                ],
            )
            self.analyzer.registry.add_recognizer(account_recognizer)

    def _load_privacy_filter(self) -> None:
        """
        Lazy-load the OpenAI Privacy Filter HuggingFace pipeline.

        Import is deferred internally to avoid loading heavy transformers packages when
        the pass is disabled.

        Optimizes execution on Apple Silicon (MPS) if available, falling back to CUDA or CPU.

        Raises:
            RuntimeError: If the transformers library or model cannot be loaded.
        """
        try:
            # Lazy import to avoid loading heavy transformers at application startup
            import torch
            from transformers import pipeline as hf_pipeline

            print(f"  * Loading OpenAI Privacy Filter from: {self.config.openai_filter_model_path}")

            # Apple Silicon GPU/MPS optimization, fallback to CUDA, then CPU (-1)
            device: str | int
            if torch.backends.mps.is_available():
                device = "mps"
                print("  * Using Apple Silicon GPU/MPS.")
            elif torch.cuda.is_available():
                device = 0
                print("  * Using NVIDIA CUDA.")
            else:
                device = -1
                print("  * Using CPU.")

            self._privacy_filter_pipeline = hf_pipeline(
                task="token-classification",
                model=self.config.openai_filter_model_path,
                aggregation_strategy="simple",
                device=device,
            )
            print("  * OpenAI Privacy Filter loaded.")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load OpenAI Privacy Filter from " + f"'{self.config.openai_filter_model_path}': {exc}"
            ) from exc

    # ── Internal masking helpers ──────────────────────────────────────────────

    def _title_case_all_caps(self, text: str) -> str:
        """Title case only words that are fully in all caps (length >= 2)."""

        def replace_match(match: re.Match[str]) -> str:
            word = match.group(0)
            if word.isalpha():
                return word.title()
            return word

        return re.sub(r"\b[A-Z]{2,}\b", replace_match, text)

    def _apply_presidio(self, text: str, session: PIISession) -> AnonymizationPassResult:
        """
        Run Presidio NER over *text* and replace detected spans with placeholders.

        Runs on both English and Italian spaCy models, using both raw text and
        Title-Cased versions to ensure high recall on Italian and all-caps entities.
        Filters out spans consisting strictly of false positive (EXCLUDE_FROM_PII_SYMBOLS).

        Args:
            text: Raw input text.
            session: Active anonymization session.

        Returns:
            AnonymizationPassResult containing anonymized text and new placeholders.
        """
        # Run 1: English (raw + selective title)
        results_en = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self._target_entities,
        )
        results_en_title = self.analyzer.analyze(
            text=self._title_case_all_caps(text),
            language="en",
            entities=self._target_entities,
        )

        # Run 2: Italian (raw + selective title)
        results_it = self.analyzer.analyze(
            text=text,
            language="it",
            entities=self._target_entities,
        )
        results_it_title = self.analyzer.analyze(
            text=self._title_case_all_caps(text),
            language="it",
            entities=self._target_entities,
        )

        # Merge results across languages/passes without overlapping indices
        combined = list(results_en) + list(results_en_title) + list(results_it) + list(results_it_title)

        # Sort key to prioritize specific, high-confidence tax identifiers (Italian Fiscal Code
        # and Irish PPSN) over generic NER entities (like PERSON, LOCATION) that might overlap.
        # Spans starting earlier are sorted first, and longer spans starting at the same index
        # come first (-r.end) to resolve nested/overlapping entities from left-to-right.
        def merge_sort_key(r: RecognizerResult) -> tuple[int, int, int]:
            priority = 0 if r.entity_type in ("ITALIAN_FISCAL_CODE", "IRISH_PPSN") else 1
            return (priority, r.start, -r.end)

        combined.sort(key=merge_sort_key)

        # Greedy interval selection: iterate through the sorted list of PII detections and
        # keep a span only if it does not overlap with any span we have already accepted.
        # High-priority tax codes and longer spans are sorted first and thus preserved.
        merged = []
        for result in combined:
            overlap = False
            for m in merged:
                if not (result.end <= m.start or result.start >= m.end):
                    overlap = True
                    break
            if not overlap:
                merged.append(result)

        # Sort right-to-left to safely modify indices
        new_placeholders = {}
        for result in sorted(merged, key=lambda r: r.start, reverse=True):
            original_val = text[result.start : result.end]

            # Date/Time false positive filter
            if DATE_TIME_REGEX.match(original_val.strip()):
                continue

            # EXCLUDE_FROM_PII_SYMBOLS filter: clean up punctuation/special chars and check.
            # Example: A generic NER model might false-positively flag table headers or financial
            # keywords like "Gross Amount" or Italian "Totale a Vs. Debito" as PII.
            # By stripping punctuation and verifying if all words are in EXCLUDE_FROM_PII_SYMBOLS,
            # we avoid anonymizing standard vocabulary terms crucial for parser logic.
            span_clean = "".join(c for c in original_val.lower() if c.isalnum() or c.isspace()).strip()
            words = span_clean.split()
            if words and all(w in EXCLUDE_FROM_PII_SYMBOLS for w in words):
                continue

            placeholder = session.make_placeholder(result.entity_type, original_val, source="PRESIDIO")
            new_placeholders[placeholder] = original_val
            text = text[: result.start] + placeholder + text[result.end :]
        return AnonymizationPassResult(text=text, placeholders=new_placeholders)

    def _write_log_file(
        self,
        log_dir: str | None,
        document_name: str | None,
        attempt: int,
        system_instruction: str,
        prompt: str,
        raw_response: str,
    ) -> None:
        """Helper to write raw LLM responses to a log file for debugging."""
        if log_dir and document_name:
            try:
                os.makedirs(log_dir, exist_ok=True)
                safe_doc_name = "".join(
                    c if c.isalnum() or c in ("-", "_", ".") else "_" for c in document_name
                )
                log_filename = os.path.join(log_dir, f"{safe_doc_name}_pii_redaction_attempt_{attempt}.txt")
                with open(log_filename, "w", encoding="utf-8") as f:
                    _ = f.write(f"=== SYSTEM PROMPT ===\n{system_instruction}\n\n")
                    _ = f.write(f"=== PROMPT ===\n{prompt}\n\n")
                    _ = f.write(f"=== RAW RESPONSE ===\n{raw_response}\n")
                logger.info("Saved raw PII redaction response log to: %s", log_filename)
            except Exception as log_err:
                logger.warning("Failed to save raw PII redaction log: %s", log_err)

    def _apply_openai_privacy_filter(self, text: str, session: PIISession) -> AnonymizationPassResult:
        """
        Run the OpenAI Privacy Filter model over *text* and replace any newly
        detected PII spans with placeholders.

        Discards matches consisting only of EXCLUDE_FROM_PII_SYMBOLS terms.

        Args:
            text: Text already processed by Presidio (may still contain PII).
            session: Active anonymization session.

        Returns:
            AnonymizationPassResult containing anonymized text and new placeholders.
        """
        if self._privacy_filter_pipeline is None:
            self._load_privacy_filter()

        pipeline = self._privacy_filter_pipeline
        if pipeline is None:
            raise RuntimeError("OpenAI Privacy Filter pipeline is None.")

        try:
            detections = pipeline(text)
        except Exception as exc:
            # Raise exception instead of failing silently to prevent PII leakage to LLMs
            raise RuntimeError(f"OpenAI Privacy Filter inference failed: {exc}") from exc

        # Filter out EXCLUDE_FROM_PII_SYMBOLS words and sort right-to-left
        detections_list = cast(list[dict[str, Any]], detections) if isinstance(detections, list) else []
        sorted_detections = sorted(detections_list, key=lambda d: cast(int, d["start"]), reverse=True)

        new_placeholders = {}
        for detection in sorted_detections:
            start: int = detection["start"]
            end: int = detection["end"]
            entity_type: str = detection["entity_group"].upper()
            original_val: str = text[start:end]

            # Date/Time false positive filter
            if DATE_TIME_REGEX.match(original_val.strip()):
                continue

            # EXCLUDE_FROM_PII_SYMBOLS filter
            span_clean = "".join(c for c in original_val.lower() if c.isalnum() or c.isspace()).strip()
            words = span_clean.split()
            if words and all(w in EXCLUDE_FROM_PII_SYMBOLS for w in words):
                continue

            placeholder = session.make_placeholder(entity_type, original_val, source="OPENAI")
            new_placeholders[placeholder] = original_val
            text = text[:start] + placeholder + text[end:]

        return AnonymizationPassResult(text=text, placeholders=new_placeholders)

    def _apply_llm_redaction(
        self,
        runner: BaseLLMRunner,
        text: str,
        session: PIISession,
        document_name: str | None = None,
        log_dir: str | None = None,
    ) -> AnonymizationPassResult:
        """
        Pass 1 — LLM-based Redaction Filter.
        Queries the configured LLM runner to redact PII from the text,
        using a structured JSON output format and updating the session placeholder mapping.
        """
        system_instruction = (
            "You are an expert data security agent.\n"
            "Your task is to redact all personally identifiable information (PII) "
            "from the provided text, while preserving all financial transactions, asset symbols, "
            "quantities, prices, and table structures.\n\n"
            "Target entities to redact (not exhaustive list):\n"
            "- PERSON: First and last names of individuals.\n"
            "- LOCATION: Complete addresses, street names, city names, and postal codes (CAP, EIRCODE, ...).\n"
            "- ACCOUNT_NUMBER: Bank account numbers, client codes, portfolio IDs, order numbers.\n"
            "- TAX_ID: Codice Fiscale, PPSN, or other national tax identifiers.\n"
            "- CONTACT: Phone numbers and email addresses.\n"
            "- ORDER NUMBERS: specific identifiers provided by the broker that are not needed to process the financial informations.\n\n"
            "You MUST NOT redact:\n"
            "- Financial tickers or asset names (e.g. AAPL, Broadcom, Franklin UCITS).\n"
            "- Numeric transaction values (e.g., prices, quantities, dates).\n"
            "- Table headers and keywords (e.g. Symbol, Quantity, Date, Sell, Debito).\n\n"
            "You must generate numbered placeholders for each redacted item following the naming pattern:\n"
            "`[ANONYMIZED_LLM_<ENTITY_TYPE>_<COUNTER>]` (e.g., [ANONYMIZED_LLM_PERSON_1], [ANONYMIZED_LLM_LOCATION_1]).\n\n"
            "The documents can be in different languages.\n"
            "Italian document might contain the document number in the form Conto XYZ, this is sentisive information.\n\n"
            "Your response must be a single, valid JSON object containing exactly two keys:\n"
            '1. "redacted_text": The complete input text, with all detected PII replaced by the placeholders.\n'
            '2. "replacements": A dictionary mapping each placeholder to the exact raw text it replaced.\n\n'
            "Example Output format:\n"
            "{\n"
            '  "redacted_text": "Document for [ANONYMIZED_LLM_PERSON_1] at [ANONYMIZED_LLM_LOCATION_1].",\n'
            '  "replacements": {\n'
            '    "[ANONYMIZED_LLM_PERSON_1]": "MARIO ROSSI",\n'
            '    "[ANONYMIZED_LLM_LOCATION_1]": "VIA ITALIANA SEGRETA 10"\n'
            "  }\n"
            "}\n"
        )

        max_attempts = 3
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = runner.complete(prompt=text, system_instruction=system_instruction)

                self._write_log_file(
                    log_dir,
                    document_name,
                    attempt,
                    system_instruction,
                    text,
                    raw_response,
                )

                cleaned_resp = BaseLLMRunner.clean_json_response(raw_response)
                data = json.loads(cleaned_resp, strict=False)

                redacted_text = data["redacted_text"]
                replacements = data["replacements"]

                # Save the replacements to the session placeholder map
                for placeholder, original in replacements.items():
                    session.placeholder_map[placeholder] = original

                return AnonymizationPassResult(text=redacted_text, placeholders=replacements)
            except Exception as exc:
                last_error = exc
                logger.warning("Attempt %d/%d for PII LLM redaction failed: %s", attempt, max_attempts, exc)

        # Fail the processing to prevent data leakage of raw PII if all attempts fail
        raise RuntimeError(
            f"LLM-based redaction pass failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text content.

        Args:
            text: The text content to compute hash from.

        Returns:
            Hex string of the SHA256 hash.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_config_flags(self) -> dict[str, bool]:
        """Get the current PII pipeline configuration flags as a dictionary.

        Returns:
            A dictionary containing configuration flags.
        """
        return {
            "presidio_enabled": bool(self.config.presidio_enabled),
            "openai_filter_enabled": bool(self.config.openai_filter_enabled),
            "llm_redaction_enabled": bool(self.config.llm_redaction),
        }

    def _read_from_cache(
        self,
        cache_file: str,
    ) -> Result[tuple[str, PIISession], ReadCacheError]:
        """Attempt to retrieve anonymized text and session state from local cache.

        Args:
            cache_file: Path to the cache file.

        Returns:
            A Result containing the cached anonymized text and populated PIISession on success, or an Exception on failure/miss.
        """
        if not os.path.exists(cache_file):
            return Failure("cache-file-not-found")

        try:
            with open(cache_file, encoding="utf-8") as f:
                cache_content = f.read()

            cached = PIICacheData.model_validate_json(cache_content)

            current_flags = self._get_config_flags()

            if cached.config_flags == current_flags:
                session = PIISession()
                session.placeholder_map = cached.placeholder_map
                return Success((cached.anonymized_text, session))

            logger.info("PII Cache miss: Configuration flags mismatch.")
            return Failure("config-file-mismatch")
        except ValueError:
            logger.warning("Failed to read from PII cache at %s: invalid cache file format", cache_file)
            return Failure("invalid-cache-file")
        except Exception as e:
            logger.debug("Failed to read from PII cache at %s: %s", cache_file, e)
            return Failure("unknown-error")

    def _write_to_cache(self, cache_file: str, text: str, placeholder_map: dict[str, str]) -> None:
        """Write anonymized text and session state to local cache.

        Args:
            cache_file: File path to save the cache data to.
            text: Anonymized text to store.
            placeholder_map: The new placeholder mappings from the current run.
        """
        if not self.config.pii_cache_enabled or not cache_file:
            return

        try:
            file_hash = os.path.basename(cache_file).replace(".json", "")

            cache_data = PIICacheData(
                hash=file_hash,
                config_flags=self._get_config_flags(),
                anonymized_text=text,
                placeholder_map=placeholder_map,
            )
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(cache_data.model_dump_json(indent=2))

            logger.info("Successfully wrote PII cache to %s", cache_file)
        except Exception as e:
            logger.warning("Failed to write PII cache to %s: %s", cache_file, e)

    # ── Public API ────────────────────────────────────────────────────────────

    def anonymize_text(
        self, text: str, force_reprocessing: bool = False, document_name: str | None = None, log_dir: str | None = None
    ) -> tuple[str, PIISession]:
        """Anonymize PII in *text* using up to three sequential filter passes.

        Pass 1 (LLM-based Redaction) runs when llm_redaction is configured.
        Pass 2 (Presidio) runs when presidio_enabled is True.
        Pass 3 (OpenAI Privacy Filter) runs when openai_filter_enabled is True.

        If all passes are disabled, the pipeline fails to initialize with a ValueError.

        Detected spans are stored in the session placeholder map for later restoration.

        Args:
            text: Raw PDF text to anonymize.
            force_reprocessing: If True, bypass the cache and run the pipeline passes.
            document_name: Optional name of the document being processed (used for logging).
            log_dir: Optional directory path to save raw LLM redaction response logs.

        Returns:
            A tuple of (anonymized_text, session).
        """
        if not text:
            return text, PIISession()

        cache_file: str | None = None
        if self.config.pii_cache_enabled:
            file_hash = self._compute_hash(text)
            cache_dir = self.config.pii_cache_dir
            cache_file = os.path.join(cache_dir, f"{file_hash}.json")

            if not force_reprocessing:
                cache_res = self._read_from_cache(cache_file)
                if isinstance(cache_res, Success):
                    logger.info("Successfully read PII cache from %s", cache_file)
                    return cache_res.unwrap()
                # This must be a failure result
                error_code = cache_res.failure()
                friendly_msg = READ_CACHE_ERROR_FRIENDLY_MESSAGES.get(error_code, "Unknown cache error.")
                logger.warning(f"PII Cache retrieval failed: {friendly_msg} (error code: {error_code})")
            else:
                logger.info("Cache miss (force_reprocessing=True), proceeding with full pipeline...")

        session = PIISession()
        run_placeholders: dict[str, str] = {}

        if self.config.llm_redaction:
            runner = self.config.llm_redaction.runner
            res = self._apply_llm_redaction(runner, text, session, document_name=document_name, log_dir=log_dir)
            text = res.text
            run_placeholders.update(res.placeholders)

        if self.config.presidio_enabled:
            res = self._apply_presidio(text, session)
            text = res.text
            run_placeholders.update(res.placeholders)

        if self.config.openai_filter_enabled:
            res = self._apply_openai_privacy_filter(text, session)
            text = res.text
            run_placeholders.update(res.placeholders)

        if cache_file:
            self._write_to_cache(cache_file, text, run_placeholders)

        return text, session

    def deanonymize_value(self, val: object, session: PIISession, raise_on_failure: bool = False) -> object:
        """
        Recursively restore placeholders in any nested string, list, or dict structure.

        Args:
            val: Nested data structure or string containing placeholders.
            session: Active anonymization session.
            raise_on_failure: If True, raise ValueError if any placeholders remain in the output.

        Returns:
            The restored structure with all placeholders replaced by original values.
        """
        if isinstance(val, str):
            restored = self.deanonymize_str(val, session)
        elif isinstance(val, list):
            restored = self.deanonymize_list(val, session)
        elif isinstance(val, dict):
            restored = self.deanonymize_dict(val, session)
        else:
            restored = val

        if raise_on_failure:
            failures = self._find_placeholders(restored)
            if failures:
                raise ValueError(f"Failed to fully deanonymize value. Remaining placeholders: {failures}")

        return restored

    def deanonymize_item(self, item: TModel, session: PIISession, raise_on_failure: bool = False) -> TModel:
        """
        Restore any anonymization placeholders found in a Pydantic model back
        to their original values using the session mapping.

        Args:
            item: Pydantic model instance containing placeholders.
            session: Active anonymization session.
            raise_on_failure: If True, raise ValueError if any placeholders remain in the output.

        Returns:
            New Pydantic model instance with placeholders restored.
        """
        if not session.placeholder_map:
            if raise_on_failure:
                failures = self._find_placeholders(item)
                if failures:
                    raise ValueError(f"Failed to fully deanonymize item. Remaining placeholders: {failures}")
            return item

        item_dict: dict[str, object] = item.model_dump()
        deanonymized_data = self.deanonymize_dict(item_dict, session)
        restored = type(item)(**deanonymized_data)

        if raise_on_failure:
            failures = self._find_placeholders(restored)
            if failures:
                raise ValueError(f"Failed to fully deanonymize item. Remaining placeholders: {failures}")

        return restored

    def _find_placeholders(self, val: object) -> list[str]:
        """
        Recursively scan a string, list, dict, or Pydantic model to find remaining anonymization placeholders.

        Args:
            val: The object to scan.

        Returns:
            A list of detected placeholder strings.
        """
        placeholders: list[str] = []
        if isinstance(val, str):
            placeholders.extend(PLACEHOLDER_REGEX.findall(val))
        elif isinstance(val, list):
            for item in val:
                placeholders.extend(self._find_placeholders(item))
        elif isinstance(val, dict):
            for v in val.values():
                placeholders.extend(self._find_placeholders(v))
        elif isinstance(val, BaseModel):
            item_dict: dict[str, object] = val.model_dump()
            placeholders.extend(self._find_placeholders(item_dict))
        return placeholders

    def deanonymize_str(self, val: str, session: PIISession) -> str:
        """Restore placeholders in a string."""
        restored = val
        for placeholder, original in session.placeholder_map.items():
            if placeholder in restored:
                restored = restored.replace(placeholder, original)
        return restored

    def deanonymize_list(self, val: list[object], session: PIISession) -> list[object]:
        """Recursively restore placeholders in a list."""
        return [self.deanonymize_value(v, session) for v in val]

    def deanonymize_dict(self, val: dict[str, object], session: PIISession) -> dict[str, object]:
        """Recursively restore placeholders in a dict."""
        return {k: self.deanonymize_value(v, session) for k, v in val.items()}
