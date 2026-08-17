from __future__ import annotations

import json as json_lib
import unittest
from datetime import datetime
from decimal import Decimal
from typing import Literal
from unittest.mock import MagicMock, patch

from backend.consensus_models import TransactionExtractionItem
from backend.domain_models import AssetType, TransactionAction
from backend.ingestion.pii.models import LLMRedactionConfig, PIIPipelineConfig
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.ingestion.pii.session import PIISession
from backend.llm.runner import BaseLLMRunner


def _make_config(
    *,
    presidio_enabled: bool = False,
    openai_filter_enabled: bool = False,
    openai_filter_model_path: str = "openai/privacy-filter",
    llm_redaction: Literal[False] | LLMRedactionConfig = False,
    pii_cache_enabled: bool = False,
    pii_cache_dir: str = ".pii_cache",
) -> PIIPipelineConfig:
    """Helper to instantiate PIIPipelineConfig with caching disabled by default in tests."""
    return PIIPipelineConfig(
        presidio_enabled=presidio_enabled,
        openai_filter_enabled=openai_filter_enabled,
        openai_filter_model_path=openai_filter_model_path,
        llm_redaction=llm_redaction,
        pii_cache_enabled=pii_cache_enabled,
        pii_cache_dir=pii_cache_dir,
    )


class TestPresidioPIIPass(unittest.TestCase):
    """Tests for Pass 1 — Presidio NER."""

    def setUp(self) -> None:
        self.pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )

    def test_fiscal_code_regex_recognizer(self) -> None:
        """Italian CF regex recognizer must always fire regardless of NER model recall."""
        text = "Tax ID CF: ROSMRI87A04H501K."
        masked, session = self.pipeline.anonymize_text(text)

        self.assertIn("[ANONYMIZED_PRESIDIO_ITALIAN_FISCAL_CODE_", masked)
        self.assertNotIn("ROSMRI87A04H501K", masked)
        self.assertIn("ROSMRI87A04H501K", session.placeholder_map.values())

    def test_ppsn_regex_recognizer(self) -> None:
        """Irish PPSN regex recognizer must always fire regardless of NER model recall."""
        text = "PPSN: 1234567FA."
        masked, session = self.pipeline.anonymize_text(text)

        self.assertIn("[ANONYMIZED_PRESIDIO_IRISH_PPSN_", masked)
        self.assertNotIn("1234567FA", masked)

    def test_organization_values_not_in_placeholder_map(self) -> None:
        """Broker and asset names must not be stored in the placeholder mapping."""
        text = "Order executed via Directa SIM for 100 shares of Broadcom Inc (AVGO)."
        masked, session = self.pipeline.anonymize_text(text)

        self.assertNotIn("Directa SIM", session.placeholder_map.values())
        self.assertNotIn("Broadcom Inc", session.placeholder_map.values())
        self.assertNotIn("AVGO", session.placeholder_map.values())

    def test_repeated_value_reuses_same_placeholder(self) -> None:
        """The same PII value appearing twice must map to the same placeholder."""
        text = "CF1: ROSMRI87A04H501K. CF2: ZNLMRC85T10H501W."
        masked, session = self.pipeline.anonymize_text(text)

        # Two different CFs → two different placeholders, counter must be 2
        self.assertIn("[ANONYMIZED_PRESIDIO_ITALIAN_FISCAL_CODE_1]", masked)
        self.assertIn("[ANONYMIZED_PRESIDIO_ITALIAN_FISCAL_CODE_2]", masked)
        self.assertEqual(session.counter.get("PRESIDIO_ITALIAN_FISCAL_CODE"), 2)

        # Re-run make_placeholder directly: values already in map, counter must not advance
        session.make_placeholder("ITALIAN_FISCAL_CODE", "ROSMRI87A04H501K", "PRESIDIO")
        self.assertEqual(session.counter.get("PRESIDIO_ITALIAN_FISCAL_CODE"), 2)

    def test_all_disabled_raises_value_error(self) -> None:
        """Initializing PIIPipeline with all passes disabled must raise ValueError."""
        with self.assertRaises(ValueError):
            PIIPipeline(_make_config(presidio_enabled=False, openai_filter_enabled=False, llm_redaction=False))

    def test_all_caps_italian_name_anonymized(self) -> None:
        """All-caps Italian names must be correctly anonymized using title-cased analysis pass."""
        text = "Document belongs to MARIO ROSSI."
        masked, session = self.pipeline.anonymize_text(text)
        self.assertIn("[ANONYMIZED_PRESIDIO_PERSON_", masked)
        self.assertNotIn("MARIO ROSSI", masked)
        self.assertIn("MARIO ROSSI", session.placeholder_map.values())

    def test_italian_address_and_cap_anonymized(self) -> None:
        """Italian address and CAP must be anonymized by custom recognizers."""
        text = "Residing at VIA SEGRETA ITALIANA 10, 40003 MILANO."
        masked, session = self.pipeline.anonymize_text(text)
        self.assertIn("[ANONYMIZED_PRESIDIO_LOCATION_1]", masked)
        self.assertIn("[ANONYMIZED_PRESIDIO_LOCATION_2]", masked)
        self.assertNotIn("VIA SEGRETA ITALIANA 10", masked)
        self.assertNotIn("40003 MILANO", masked)

    def test_whitelisted_words_not_anonymized(self) -> None:
        """Whitelisted financial terms and table headers must not be anonymized."""
        text = "You can Sell, Description: Gross Amount, Totale a Vs. Debito."
        masked, session = self.pipeline.anonymize_text(text)
        self.assertEqual(masked, text)
        self.assertEqual(len(session.placeholder_map), 0)

    def test_account_number_anonymized(self) -> None:
        """Presidio must anonymize account numbers matching custom recognizers."""
        pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )
        text = "Order executed under Conto FE765 for account U***12123."
        masked, session = pipeline.anonymize_text(text)

        self.assertIn("[ANONYMIZED_PRESIDIO_ACCOUNT_NUMBER_1]", masked)
        self.assertIn("[ANONYMIZED_PRESIDIO_ACCOUNT_NUMBER_2]", masked)
        self.assertNotIn("FE765", masked)
        self.assertNotIn("U***12123", masked)

    def test_date_time_not_anonymized_as_location(self) -> None:
        """Dates and timestamps must not be falsely anonymized as LOCATION/PERSON."""
        pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )
        text = "Executed del 4.07.2024 at 16:35:55, valuta 08.07.2024."
        masked, session = pipeline.anonymize_text(text)

        self.assertEqual(masked, text)
        self.assertEqual(len(session.placeholder_map), 0)


class TestOpenAIPrivacyFilterPass(unittest.TestCase):
    """Tests for Pass 2 — OpenAI Privacy Filter local model."""

    def _make_pipeline(self, mock_detections: list[dict[str, object]]) -> PIIPipeline:
        """Create a PIIPipeline with a mocked HF token-classification pipeline."""
        pii = PIIPipeline(_make_config(presidio_enabled=False, openai_filter_enabled=True, llm_redaction=False))

        mock_hf_pipeline = MagicMock(return_value=mock_detections)
        pii._privacy_filter_pipeline = mock_hf_pipeline
        return pii

    def test_second_pass_detects_residual_pii(self) -> None:
        """Privacy filter detections must be replaced with ANONYMIZED_ placeholders."""
        text = "Signed by: Giovanni SonoVero, broker: Directa."
        start = text.index("Giovanni SonoVero")
        end = start + len("Giovanni SonoVero")
        detections: list[dict[str, object]] = [
            {"entity_group": "private_person", "start": start, "end": end, "score": 0.99},
        ]
        pipeline = self._make_pipeline(detections)
        masked, session = pipeline.anonymize_text(text)

        self.assertIn("[ANONYMIZED_OPENAI_PRIVATE_PERSON_", masked)
        self.assertNotIn("Giovanni SonoVero", masked)
        self.assertIn("Giovanni SonoVero", session.placeholder_map.values())

    def test_second_pass_labels_uppercased(self) -> None:
        """Entity label from model (e.g. 'private_address') must be uppercased in placeholder."""
        detections: list[dict[str, object]] = [
            {"entity_group": "private_address", "start": 3, "end": 28, "score": 0.97},
        ]
        text = "at VIA SEGRETA ITALIANA 10, Milan."
        pipeline = self._make_pipeline(detections)
        masked, session = pipeline.anonymize_text(text)

        keys = list(session.placeholder_map.keys())
        self.assertTrue(any("PRIVATE_ADDRESS" in k for k in keys))
        self.assertNotIn("VIA SEGRETA ITALIANA 10", masked)

    def test_second_pass_preserves_non_pii_text(self) -> None:
        """Text segments not flagged by the model must remain untouched."""
        detections: list[dict[str, object]] = [
            {"entity_group": "private_person", "start": 0, "end": 12, "score": 0.99},
        ]
        text = "Alice Smith bought AAPL shares."
        pipeline = self._make_pipeline(detections)
        masked, session = pipeline.anonymize_text(text)

        self.assertIn("bought AAPL shares", masked)

    def test_second_pass_disabled_skips_model(self) -> None:
        """When openai_filter_enabled=False, the HF pipeline must never be loaded or called."""
        pii = PIIPipeline(_make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False))
        with patch.object(pii, "_load_privacy_filter") as mock_load:
            pii.anonymize_text("Alice Smith at 123 Main St.")
            mock_load.assert_not_called()

    def test_inference_failure_raises_exception(self) -> None:
        """A crash in HF inference must raise a RuntimeError rather than leaking raw PII."""
        pii = PIIPipeline(_make_config(presidio_enabled=False, openai_filter_enabled=True, llm_redaction=False))
        pii._privacy_filter_pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))

        text = "CF: ROSMRI87A04H501K"
        with self.assertRaises(RuntimeError):
            pii.anonymize_text(text)


class TestDeanonymization(unittest.TestCase):
    """Tests for placeholder restoration via deanonymize_item."""

    def setUp(self) -> None:
        self.pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )
        self.session = PIISession()
        # Manually seed the mapping so tests don't depend on NER recall
        self.session.placeholder_map["[ANONYMIZED_PRESIDIO_PERSON_1]"] = "Giovanni SonoVero"
        self.session.placeholder_map["[ANONYMIZED_OPENAI_PRIVATE_PERSON_1]"] = "Alice Smith"
        self.session.counter = {"PRESIDIO_PERSON": 1, "OPENAI_PRIVATE_PERSON": 1}

    def _make_item(
        self,
        *,
        event_date: datetime = datetime(2025, 6, 15, 15, 45, 0),
        asset_type: AssetType = AssetType.STOCK,
        symbol: str | None = "AAPL",
        action: TransactionAction = TransactionAction.BUY,
        total_amount: Decimal | float = Decimal("1500.0"),
        provider: str | None = None,
    ) -> TransactionExtractionItem:
        return TransactionExtractionItem(
            event_date=event_date,
            asset_type=asset_type,
            symbol=symbol,
            action=action,
            total_amount=Decimal(str(total_amount)) if not isinstance(total_amount, Decimal) else total_amount,
            provider=provider,
        )

    def test_presidio_placeholder_restored(self) -> None:
        item = self._make_item(provider="[ANONYMIZED_PRESIDIO_PERSON_1]")
        restored = self.pipeline.deanonymize_item(item, self.session)
        self.assertEqual(restored.provider, "Giovanni SonoVero")

    def test_privacy_filter_placeholder_restored(self) -> None:
        item = self._make_item(provider="[ANONYMIZED_OPENAI_PRIVATE_PERSON_1]")
        restored = self.pipeline.deanonymize_item(item, self.session)
        self.assertEqual(restored.provider, "Alice Smith")

    def test_non_placeholder_fields_untouched(self) -> None:
        item = self._make_item(symbol="AAPL", provider="Directa")
        restored = self.pipeline.deanonymize_item(item, self.session)
        self.assertEqual(restored.symbol, "AAPL")
        self.assertEqual(restored.provider, "Directa")

    def test_deanonymize_works_even_when_anonymizer_disabled(self) -> None:
        """deanonymize_item must still restore values if mapping exists, even if passes are disabled."""
        pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )
        item = self._make_item(provider="[ANONYMIZED_PRESIDIO_PERSON_1]")
        restored = pipeline.deanonymize_item(item, self.session)
        self.assertEqual(restored.provider, "Giovanni SonoVero")


class TestSessionManagement(unittest.TestCase):
    """Tests for session isolation via separate session objects."""

    def test_reset_clears_placeholder_map_and_counters(self) -> None:
        session = PIISession()
        session.placeholder_map["[ANONYMIZED_PRESIDIO_PERSON_1]"] = "Alice"
        session.counter["PRESIDIO_PERSON"] = 1

        session.reset()

        self.assertEqual(len(session.placeholder_map), 0)
        self.assertEqual(len(session.counter), 0)

    def test_separate_sessions_prevent_bleed(self) -> None:
        """Placeholder allocated in doc 1 must not appear in doc 2 using a separate session."""
        pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )

        masked1, session1 = pipeline.anonymize_text("CF: ROSMRI87A04H501K.")
        self.assertEqual(session1.counter.get("PRESIDIO_ITALIAN_FISCAL_CODE"), 1)

        masked2, session2 = pipeline.anonymize_text("CF: ZNLMRC85T10H501W.")

        # In new session, counter must start from 1
        self.assertEqual(session2.counter.get("PRESIDIO_ITALIAN_FISCAL_CODE"), 1)
        self.assertNotIn("ROSMRI87A04H501K", session2.placeholder_map.values())


class TestPIIValidation(unittest.TestCase):
    """Tests for validating deanonymization completion and surfacing failures."""

    def setUp(self) -> None:
        self.pipeline = PIIPipeline(
            _make_config(presidio_enabled=True, openai_filter_enabled=False, llm_redaction=False)
        )
        self.session = PIISession()
        self.session.placeholder_map["[ANONYMIZED_PRESIDIO_PERSON_1]"] = "Alice"

    def _make_item(
        self,
        *,
        event_date: datetime = datetime(2025, 6, 15, 15, 45, 0),
        asset_type: AssetType = AssetType.STOCK,
        symbol: str | None = "AAPL",
        action: TransactionAction = TransactionAction.BUY,
        total_amount: Decimal | float = Decimal("1500.0"),
        provider: str | None = None,
    ) -> TransactionExtractionItem:
        return TransactionExtractionItem(
            event_date=event_date,
            asset_type=asset_type,
            symbol=symbol,
            action=action,
            total_amount=Decimal(str(total_amount)) if not isinstance(total_amount, Decimal) else total_amount,
            provider=provider,
        )

    def test_raise_on_failure_value(self) -> None:
        """deanonymize_value must raise ValueError if any placeholders remain when raise_on_failure=True."""
        text = "Hello [ANONYMIZED_PRESIDIO_PERSON_1] and [ANONYMIZED_OPENAI_PERSON_2]."

        # Without raise_on_failure: partial mapping does not raise
        restored = self.pipeline.deanonymize_value(text, self.session, raise_on_failure=False)
        self.assertIn("[ANONYMIZED_OPENAI_PERSON_2]", str(restored))

        # With raise_on_failure: raises ValueError showing un-deanonymized placeholders
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.deanonymize_value(text, self.session, raise_on_failure=True)
        self.assertIn("[ANONYMIZED_OPENAI_PERSON_2]", str(ctx.exception))

    def test_raise_on_failure_item(self) -> None:
        """deanonymize_item must raise ValueError if placeholders remain when raise_on_failure=True."""
        item = self._make_item(provider="[ANONYMIZED_OPENAI_PERSON_2]")

        with self.assertRaises(ValueError) as ctx:
            self.pipeline.deanonymize_item(item, self.session, raise_on_failure=True)
        self.assertIn("[ANONYMIZED_OPENAI_PERSON_2]", str(ctx.exception))


class TestLLMRedactionPass(unittest.TestCase):
    """Tests for Pass 3 — LLM-based Redaction."""

    def test_llm_redaction_anonymizes_and_maps(self) -> None:
        """Pass 3 must query the runner, mask targeted info, and update placeholders."""
        mock_runner = MagicMock(spec=BaseLLMRunner)
        mock_runner.complete.return_value = json_lib.dumps(
            {
                "redacted_text": "Signed by [ANONYMIZED_LLM_PERSON_1] at [ANONYMIZED_LLM_LOCATION_1].",
                "replacements": {
                    "[ANONYMIZED_LLM_PERSON_1]": "MARIO ROSSI",
                    "[ANONYMIZED_LLM_LOCATION_1]": "VIA SEGRETA ITALIANA 10",
                },
            }
        )

        pipeline = PIIPipeline(
            config=_make_config(
                presidio_enabled=False,
                openai_filter_enabled=False,
                llm_redaction=LLMRedactionConfig(runner=mock_runner),
            )
        )
        text = "Signed by MARIO ROSSI at VIA SEGRETA ITALIANA 10."
        masked, session = pipeline.anonymize_text(text)

        self.assertIn("[ANONYMIZED_LLM_PERSON_1]", masked)
        self.assertIn("[ANONYMIZED_LLM_LOCATION_1]", masked)
        self.assertNotIn("MARIO ROSSI", masked)
        self.assertEqual(session.placeholder_map["[ANONYMIZED_LLM_PERSON_1]"], "MARIO ROSSI")
        self.assertEqual(session.placeholder_map["[ANONYMIZED_LLM_LOCATION_1]"], "VIA SEGRETA ITALIANA 10")

    def test_llm_redaction_handles_think_blocks(self) -> None:
        """Pass 3 must strip reasoning/think blocks (like DeepSeek-R1 outputs) before parsing."""
        mock_runner = MagicMock(spec=BaseLLMRunner)
        mock_runner.complete.return_value = (
            "<think>\nSome reasoning chain...\n</think>\n"
            "{\n"
            '  "redacted_text": "Hi [ANONYMIZED_LLM_PERSON_1]",\n'
            '  "replacements": {"[ANONYMIZED_LLM_PERSON_1]": "Giovanni"}\n'
            "}"
        )

        pipeline = PIIPipeline(
            config=_make_config(
                presidio_enabled=False,
                openai_filter_enabled=False,
                llm_redaction=LLMRedactionConfig(runner=mock_runner),
            )
        )
        masked, session = pipeline.anonymize_text("Hi Giovanni")
        self.assertEqual(masked, "Hi [ANONYMIZED_LLM_PERSON_1]")

    def test_llm_redaction_failure_raises_exception(self) -> None:
        """Pass 1 must raise a RuntimeError if the LLM runner throws an exception."""
        mock_runner = MagicMock(spec=BaseLLMRunner)
        mock_runner.complete.side_effect = RuntimeError("Connection timeout")

        pipeline = PIIPipeline(
            config=_make_config(
                presidio_enabled=False,
                openai_filter_enabled=False,
                llm_redaction=LLMRedactionConfig(runner=mock_runner),
            )
        )
        with self.assertRaises(RuntimeError):
            pipeline.anonymize_text("some text")
