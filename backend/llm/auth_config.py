"""Authentication configuration models for LLM runners and runner factories."""

from typing import Literal

from pydantic import BaseModel


class APIKeyAuthConfig(BaseModel):
    """Configuration for API Key based authentication."""

    auth_type: Literal["API_KEY"] = "API_KEY"
    api_key: str


class GCPADCAuthConfig(BaseModel):
    """Configuration for GCP Application Default Credentials based authentication."""

    auth_type: Literal["GCP_ADC"] = "GCP_ADC"


class NoAuthConfig(BaseModel):
    """Configuration for unauthenticated endpoints (e.g. local vLLM / Ollama)."""

    auth_type: Literal["NONE"] = "NONE"


LLMAuthConfig = APIKeyAuthConfig | GCPADCAuthConfig | NoAuthConfig
