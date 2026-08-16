from pathlib import Path

from backend.ingestion.pii.models import AnonymizationPassResult, PIIPipelineConfig
from backend.ingestion.pii.pii_pipeline import PIIPipeline
from backend.ingestion.pii.session import PIISession


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

    # When: Anonymizing the text for the first time
    masked_1, session_1 = pipeline.anonymize_text(text)

    # Then: It should be a cache miss, anonymization occurs, and cache file is written
    assert "[ANONYMIZED_PRESIDIO_ITALIAN_FISCAL_CODE_" in masked_1
    assert "ROSMRI87A04H501K" in session_1.placeholder_map.values()

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1

    # TODO: Without spying either the anonymizer or the file, this test doesn't prove much.
    # When: Anonymizing the same text with a new session (triggering cache hit)
    masked_2, session_2 = pipeline.anonymize_text(text)

    # Then: It should be a cache hit, restoring session and returning identical text
    assert masked_2 == masked_1
    assert session_2.placeholder_map == session_1.placeholder_map


def test_pii_cache_config_flags_mismatch(tmp_path: Path) -> None:
    # Given: A cache entry created under a specific configuration
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

    # When: Running again with a different config configuration (e.g. presidio disabled)
    config_2 = PIIPipelineConfig(
        presidio_enabled=False,
        openai_filter_enabled=True,
        # mock path to prevent heavy downloads
        openai_filter_model_path="tests/mock_model",
        llm_redaction=False,
        pii_cache_enabled=True,
        pii_cache_dir=str(tmp_path),
    )
    # TODO: Are we sure that with this mocking we are hitting the right code path ?
    # TODO: Is the mocked _apply_openai_privacy_filter actually used ? Is there a way
    # to only mock the dependency and not the private method (that might change) ?

    # We mock loading of the privacy filter to prevent heavy downloads
    pipeline_2 = PIIPipeline(config=config_1)  # initialized first
    pipeline_2.config = config_2  # swap config to mimic changes

    # We also mock the _apply_openai_privacy_filter method
    def mock_privacy_filter(text: str, session: PIISession) -> AnonymizationPassResult:
        placeholder = session.make_placeholder("MOCK_ENT", "ROSMRI87A04H501K", "MOCK")
        return AnonymizationPassResult(
            text=text.replace("ROSMRI87A04H501K", placeholder), placeholders={placeholder: "ROSMRI87A04H501K"}
        )

    pipeline_2._apply_openai_privacy_filter = mock_privacy_filter

    masked_2, session_2 = pipeline_2.anonymize_text(text)

    # TODO: Without spying on the cache file (E.g. not read), we are not really sure that the cache wasn't read.
    # Then: The cache should be ignored due to config mismatch, triggering a new run
    assert "[ANONYMIZED_MOCK_MOCK_ENT_" in masked_2
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

    # When: Running again with force_reprocessing=True
    masked_2, session_2 = pipeline.anonymize_text(text, force_reprocessing=True)

    # TODO: This assertion does not validate that the processing has been reforced. We need to spy
    # Then: It should ignore the cache and re-process, populating the new session
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

    # When: Anonymizing text_1
    pipeline.anonymize_text(text_1)

    # Then: Cache file for text_1 hash exists
    hash_1 = pipeline._compute_hash(text_1)
    assert (tmp_path / f"{hash_1}.json").exists()

    # When: Anonymizing text_2
    pipeline.anonymize_text(text_2)

    # Then: Cache file for text_2 hash exists, and is different
    hash_2 = pipeline._compute_hash(text_2)
    assert hash_1 != hash_2
    assert (tmp_path / f"{hash_2}.json").exists()


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
