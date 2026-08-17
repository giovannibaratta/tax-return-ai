"""Tests for local disk caching of PII anonymization results and sessions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.ingestion.pii.models import LLMRedactionConfig, PIIPipelineConfig
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.llm.runner import BaseLLMRunner


def test_pii_cache_hit_and_miss(tmp_path: Path) -> None:
    # Given: A PII pipeline configured to use a temporary cache directory
    config = PIIPipelineConfig(
        presidio_enabled=True,
        openai_filter_enabled=False,
        llm_redaction=False,
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    pipeline = PIIPipeline(config=config)
    text = "Tax ID: ROSMRI87A04H501K."

    # When: Anonymizing the text for the first time (spy on public AnalyzerEngine dependency)
    with patch.object(pipeline.analyzer, "analyze", wraps=pipeline.analyzer.analyze) as mock_analyze:
        masked_1, session_1 = pipeline.anonymize_text(text)

        # Then: It should be a cache miss, executing analyzer passes and writing cache file
        miss_call_count = mock_analyze.call_count
        assert miss_call_count > 0
        assert "[ANONYMIZED_PRESIDIO_ITALIAN_FISCAL_CODE_" in masked_1
        assert "ROSMRI87A04H501K" in session_1.placeholder_map.values()

        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

        # When: Anonymizing the same text with a new session (triggering cache hit)
        masked_2, session_2 = pipeline.anonymize_text(text)

        # Then: Cache hit without re-invoking the analyzer dependency
        assert mock_analyze.call_count == miss_call_count
        assert masked_2 == masked_1
        assert session_2.placeholder_map == session_1.placeholder_map


def test_pii_cache_config_flags_mismatch(tmp_path: Path) -> None:
    # Given: Cache entry created under a presidio-only configuration
    config_1 = PIIPipelineConfig(
        presidio_enabled=True,
        openai_filter_enabled=False,
        llm_redaction=False,
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    pipeline_1 = PIIPipeline(config=config_1)
    text = "Tax ID: ROSMRI87A04H501K."
    pipeline_1.anonymize_text(text)

    # When: Running with a different configuration using an injected LLM runner dependency
    mock_runner = MagicMock(spec=BaseLLMRunner)
    mock_runner.complete.return_value = (
        '{"redacted_text": "Tax ID: [ANONYMIZED_LLM_TAX_ID_1].", '
        '"replacements": {"[ANONYMIZED_LLM_TAX_ID_1]": "ROSMRI87A04H501K"}}'
    )
    config_2 = PIIPipelineConfig(
        presidio_enabled=False,
        openai_filter_enabled=False,
        llm_redaction=LLMRedactionConfig(runner=mock_runner),
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    pipeline_2 = PIIPipeline(config=config_2)

    masked_2, session_2 = pipeline_2.anonymize_text(text)

    # Then: Cache is bypassed due to configuration mismatch and injected runner executes
    assert mock_runner.complete.call_count == 1
    assert "[ANONYMIZED_LLM_TAX_ID_1]" in masked_2
    assert "ROSMRI87A04H501K" in session_2.placeholder_map.values()


def test_pii_cache_force_reprocessing(tmp_path: Path) -> None:
    # Given: A cache entry already exists
    config = PIIPipelineConfig(
        presidio_enabled=True,
        openai_filter_enabled=False,
        llm_redaction=False,
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    pipeline = PIIPipeline(config=config)
    text = "Tax ID: ROSMRI87A04H501K."
    pipeline.anonymize_text(text)

    # When: Running again with force_reprocessing=True (spying on public analyzer dependency)
    with patch.object(pipeline.analyzer, "analyze", wraps=pipeline.analyzer.analyze) as mock_analyze:
        masked_2, session_2 = pipeline.anonymize_text(text, force_reprocessing=True)

        # Then: Bypasses cache and calls analyzer
        assert mock_analyze.call_count > 0
        assert "[ANONYMIZED_PRESIDIO_ITALIAN_FISCAL_CODE_" in masked_2
        assert "ROSMRI87A04H501K" in session_2.placeholder_map.values()


def test_pii_cache_text_change(tmp_path: Path) -> None:
    # Given: A PII pipeline and two different texts
    config = PIIPipelineConfig(
        presidio_enabled=True,
        openai_filter_enabled=False,
        llm_redaction=False,
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    pipeline = PIIPipeline(config=config)
    text_1 = "Tax ID: ROSMRI87A04H501K."
    text_2 = "Tax ID: ZNLMRC85T10H501W."

    # When: Anonymizing both texts
    pipeline.anonymize_text(text_1)
    pipeline.anonymize_text(text_2)

    # Then: Two distinct cache files are created on disk
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 2


def test_pii_cache_isolation(tmp_path: Path) -> None:
    # Given: A PII pipeline
    config = PIIPipelineConfig(
        presidio_enabled=True,
        openai_filter_enabled=False,
        llm_redaction=False,
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    pipeline = PIIPipeline(config=config)

    text_1 = "Tax ID: ROSMRI87A04H501K."
    text_2 = "Tax ID: ZNLMRC85T10H501W."

    # When: Anonymizing both texts
    masked_1, session_1 = pipeline.anonymize_text(text_1)
    masked_2, session_2 = pipeline.anonymize_text(text_2)

    # Then: The returned sessions are completely isolated
    assert "ROSMRI87A04H501K" in session_1.placeholder_map.values()
    assert "ZNLMRC85T10H501W" not in session_1.placeholder_map.values()

    assert "ZNLMRC85T10H501W" in session_2.placeholder_map.values()
    assert "ROSMRI87A04H501K" not in session_2.placeholder_map.values()
