import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.llm.runner import BaseLLMRunner
from backend.llm.runner_factory import build_runner


class LLMRedactionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    runner: BaseLLMRunner

    @classmethod
    def from_env(cls) -> "LLMRedactionConfig":
        """Load configuration explicitly from environment variables."""
        runner = build_runner("PII")
        return cls(runner=runner)


class PIIPipelineConfig(BaseModel):
    """
    Configuration schema for the PII Anonymization and De-anonymization pipeline.

    Reads defaults from environment variables to keep configuration decoupled from logic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    presidio_enabled: bool = Field(default=False, description="Whether Presidio NER is active.")
    openai_filter_enabled: bool = Field(default=False, description="Whether OpenAI Privacy Filter is active.")
    openai_filter_model_path: str = Field(
        default="openai/privacy-filter", description="HuggingFace model repo id or local model directory path."
    )

    llm_redaction: Literal[False] | LLMRedactionConfig = Field(
        description="Whether LLM-based Redaction Filter is active."
    )

    pii_cache_enabled: bool = Field(default=True, description="Whether PII caching is active.")
    pii_cache_dir: str = Field(default=".pii_cache", description="Directory for local PII caching.")

    @classmethod
    def from_env(cls) -> "PIIPipelineConfig":
        """Load configuration explicitly from environment variables."""

        presidio_enabled = os.environ.get("PII_PRESIDIO_ENABLED", "false").lower() in ("true", "1", "yes")
        openai_filter_enabled = os.environ.get("OPENAI_PRIVACY_FILTER_ENABLED", "false").lower() in ("true", "1", "yes")
        openai_filter_model_path = os.environ.get("OPENAI_PRIVACY_FILTER_MODEL_PATH", "openai/privacy-filter")
        llm_redaction_enabled = os.environ.get("PII_LLM_REDACTION_ENABLED", "true").lower() in ("true", "1", "yes")

        pii_cache_enabled = os.environ.get("PII_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        pii_cache_dir = os.environ.get("PII_CACHE_DIR", ".pii_cache")

        llm_redaction: LLMRedactionConfig | Literal[False] = False

        if llm_redaction_enabled:
            llm_redaction = LLMRedactionConfig.from_env()

        return cls(
            presidio_enabled=presidio_enabled,
            openai_filter_enabled=openai_filter_enabled,
            openai_filter_model_path=openai_filter_model_path,
            llm_redaction=llm_redaction,
            pii_cache_enabled=pii_cache_enabled,
            pii_cache_dir=pii_cache_dir,
        )


class PIICacheData(BaseModel):
    """Schema for cached anonymized PII pipeline results on the local filesystem."""

    hash: str
    config_flags: dict[str, bool]
    anonymized_text: str
    placeholder_map: dict[str, str]


class AnonymizationPassResult(BaseModel):
    """Result of a single anonymization pass containing the redacted text and new placeholders."""

    text: str
    placeholders: dict[str, str]


ReadCacheError = Literal["config-file-mismatch", "cache-file-not-found", "invalid-cache-file", "unknown-error"]
