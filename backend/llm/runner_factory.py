import os
from typing import cast

import google.auth  # type: ignore[import-untyped]
import google.auth.credentials  # type: ignore[import-untyped]
import google.auth.transport.requests
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.providers.openai import OpenAIProvider

from backend.llm.auth_config import (
    APIKeyAuthConfig,
    GCPADCAuthConfig,
    LLMAuthConfig,
    NoAuthConfig,
)
from backend.llm.pydantic_ai_runner import PydanticAIRunner
from backend.llm.runner import BaseLLMRunner
from backend.llm.runners import OpenAICompatibleRunner, VertexAIRunner


def build_runner(prefix: str) -> BaseLLMRunner:
    """Build a BaseLLMRunner based on the provider config prefix.

    Args:
        prefix: Env var prefix, e.g. "VOTER_1", "PII", "DEBATE_LLM", "DEFAULT"

    Returns:
        An instance of BaseLLMRunner (either OpenAICompatibleRunner or VertexAIRunner).

    Raises:
        ValueError: If configuration is incomplete or provider is unknown.
    """
    # 1. Provider
    default_provider = os.environ.get("DEFAULT_LLM_PROVIDER")
    if not default_provider:
        raise ValueError(
            "DEFAULT_LLM_PROVIDER environment variable is not set. " + "A default provider must always be configured."
        )

    provider = os.environ.get(f"{prefix}_PROVIDER")
    if not provider:
        provider = default_provider

    provider = provider.strip().lower()

    # 2. Model Name
    default_model = os.environ.get("DEFAULT_LLM_MODEL")
    if not default_model:
        raise ValueError(
            "DEFAULT_LLM_MODEL environment variable is not set. " + "A default model must always be configured."
        )

    model_name = os.environ.get(f"{prefix}_MODEL")
    if not model_name:
        model_name = default_model

    #  3. Provider
    if provider == "vertex":
        return _parse_vertex_config(prefix, model_name)

    if provider == "openai-compatible":
        return _parse_openai_config(prefix, model_name)

    raise ValueError(f"Unknown provider '{provider}' configured for prefix '{prefix}'.")


def _parse_vertex_config(prefix: str, model_name: str) -> BaseLLMRunner:
    location = os.environ.get(f"{prefix}_LOCATION") or os.environ.get("DEFAULT_GCP_LOCATION")

    return VertexAIRunner(model_name=model_name, location=location)


def _parse_openai_config(prefix: str, model_name: str) -> BaseLLMRunner:
    base_url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("DEFAULT_LLM_BASE_URL")

    if not base_url:
        raise ValueError(
            f"{prefix}_BASE_URL and DEFAULT_LLM_BASE_URL are both not set. "
            + "At least one must be configured for openai-compatible provider."
        )

    # Read auth type
    auth_type = os.environ.get(f"{prefix}_AUTH_TYPE") or os.environ.get("DEFAULT_LLM_AUTH_TYPE")

    if not auth_type:
        raise ValueError(
            f"{prefix}_AUTH_TYPE and DEFAULT_LLM_AUTH_TYPE are both not set. "
            + "At least one must be configured for openai-compatible provider."
        )

    auth_type_str = auth_type.strip().upper()

    if auth_type_str == "NONE":
        runner_auth_config: LLMAuthConfig = NoAuthConfig()
    elif auth_type_str == "GCP_ADC":
        runner_auth_config = GCPADCAuthConfig()
    elif auth_type_str == "API_KEY":
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("DEFAULT_LLM_API_KEY")
        if not api_key:
            raise ValueError(f"API key is required for API_KEY auth type under prefix '{prefix}'.")
        runner_auth_config = APIKeyAuthConfig(api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported auth type '{auth_type_str}' for provider 'openai-compatible' under prefix '{prefix}'."
        )

    return OpenAICompatibleRunner(model_name=model_name, base_url=base_url, auth_config=runner_auth_config)


def build_pydantic_model(prefix: str) -> Model:
    """Build a PydanticAI Model from the provider config prefix.

    Reads the same env-var conventions as ``build_runner`` so existing
    deployment configs work without changes:

    - ``{PREFIX}_PROVIDER`` / ``DEFAULT_LLM_PROVIDER``: ``"openai-compatible"`` or ``"vertex"``
    - ``{PREFIX}_MODEL`` / ``DEFAULT_LLM_MODEL``: model name string
    - OpenAI-compatible: ``{PREFIX}_BASE_URL``, ``{PREFIX}_AUTH_TYPE``, ``{PREFIX}_API_KEY``
    - Vertex: ``{PREFIX}_LOCATION`` / ``DEFAULT_GCP_LOCATION``, ``GCP_PROJECT_ID``

    Args:
        prefix: Env var prefix, e.g. ``"PLAINTIFF"``, ``"DEFENSE"``, ``"JUDGE"``.

    Returns:
        A PydanticAI ``Model`` instance (``OpenAIChatModel`` backed by OpenAI-compatible
        or Google Cloud Vertex provider).

    Raises:
        ValueError: If required configuration is missing or provider is unknown.
    """
    default_provider = os.environ.get("DEFAULT_LLM_PROVIDER")
    if not default_provider:
        raise ValueError("DEFAULT_LLM_PROVIDER environment variable is not set.")

    provider = (os.environ.get(f"{prefix}_PROVIDER") or default_provider).strip().lower()

    default_model = os.environ.get("DEFAULT_LLM_MODEL")
    if not default_model:
        raise ValueError("DEFAULT_LLM_MODEL environment variable is not set.")

    model_name = os.environ.get(f"{prefix}_MODEL") or default_model

    if provider == "openai-compatible":
        return _build_pydantic_openai_model(prefix, model_name)

    if provider == "vertex":
        return _build_pydantic_vertex_model(prefix, model_name)

    raise ValueError(f"Unknown provider '{provider}' for prefix '{prefix}'. Supported: 'openai-compatible', 'vertex'.")


def _build_pydantic_openai_model(prefix: str, model_name: str) -> Model:
    """Build a PydanticAI OpenAIChatModel instance for OpenAI-compatible providers."""

    default_base_url = os.environ.get("DEFAULT_LLM_BASE_URL")
    base_url = os.environ.get(f"{prefix}_BASE_URL") or default_base_url
    if not base_url:
        raise ValueError(
            f"Neither {prefix}_BASE_URL nor DEFAULT_LLM_BASE_URL is available. "
            + "BASE_URL is required for openai-compatible provider."
        )

    auth_type = (os.environ.get(f"{prefix}_AUTH_TYPE") or os.environ.get("DEFAULT_LLM_AUTH_TYPE") or "").upper()
    api_key = _resolve_pydantic_api_key(prefix, auth_type)

    base_url = base_url.rstrip("/")
    if "v1" not in base_url.lower() and "chat/completions" not in base_url.lower():
        base_url = f"{base_url}/v1"

    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


def _resolve_pydantic_api_key(prefix: str, auth_type: str) -> str:
    """Resolve API key or GCP ADC access token for Pydantic OpenAIProvider."""
    if auth_type == "NONE":
        return "NONE"

    if auth_type == "API_KEY":
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("DEFAULT_LLM_API_KEY")
        if not api_key:
            raise ValueError(f"API key required for API_KEY auth under prefix '{prefix}'.")
        return api_key

    if auth_type == "GCP_ADC":
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        auth_request = google.auth.transport.requests.Request()  # pyright: ignore[reportUnknownMemberType]
        creds.refresh(auth_request)  # type: ignore[no-untyped-call] # pyright: ignore[reportUnknownMemberType]
        creds_typed = cast(google.auth.credentials.Credentials, creds)
        token = creds_typed.token  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(token, str):
            raise ValueError("GCP ADC token is not a string.")
        return token

    raise ValueError(f"Unsupported auth type '{auth_type}' for openai-compatible under prefix '{prefix}'.")


def _build_pydantic_vertex_model(prefix: str, model_name: str) -> Model:
    """Build a PydanticAI GoogleModel instance targeting Vertex AI."""
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID is required for vertex provider.")

    location = os.environ.get(f"{prefix}_LOCATION") or os.environ.get("DEFAULT_GCP_LOCATION")
    return GoogleModel(
        model_name,
        provider=GoogleCloudProvider(project=project_id, location=location),
    )


def build_pydantic_ai_runner(prefix: str) -> PydanticAIRunner:
    """Build a PydanticAIRunner using configured provider and model credentials under given prefix.

    Args:
        prefix: Env var prefix, e.g. "VOTER_1", "VOTER_2", "VOTER_3", "DEFAULT".

    Returns:
        Configured PydanticAIRunner instance.
    """
    model = build_pydantic_model(prefix)
    return PydanticAIRunner(model=model)
