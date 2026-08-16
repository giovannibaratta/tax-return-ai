"""LLM Runners module providing implementations for OpenAI-compatible, Vertex AI, and Mock LLM providers."""

import os
import random
import sys
import time
from collections.abc import Callable
from typing import TypeVar, cast
from warnings import deprecated

import google.auth
import google.auth.credentials
import google.auth.transport.requests
import requests
import vertexai
from google.api_core.exceptions import (
    ResourceExhausted,
    TooManyRequests,
)
from google.auth.credentials import with_scopes_if_required  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerativeModel

from backend.llm.auth_config import (
    APIKeyAuthConfig,
    GCPADCAuthConfig,
    LLMAuthConfig,
    NoAuthConfig,
)
from backend.llm.interaction_logger import log_request_start, log_response_finish
from backend.llm.mock_data import get_mock_pii_json, get_mock_voter_json
from backend.llm.runner import BaseLLMRunner


class _RateLimitError(RuntimeError):
    """Internal exception to signify rate limit (HTTP 429) errors."""


class OpenAIChatMessage(BaseModel):
    """Single message payload in OpenAI Chat Completion API."""

    role: str
    content: str


class OpenAIChatPayload(BaseModel):
    """Payload for OpenAI Chat Completion API request.

    Note:
        Temperature is set to 0.2 to ensure deterministic JSON structured outputs
        for tax calculation and record parsing pipelines.
    """

    model: str
    messages: list[OpenAIChatMessage]
    temperature: float = Field(default=0.2, description="Low temperature for deterministic tax data extraction")


class OpenAIChatChoiceMessage(BaseModel):
    """Message object inside an OpenAI response choice."""

    content: str


class OpenAIChatChoice(BaseModel):
    """Choice item inside OpenAI chat completion response."""

    message: OpenAIChatChoiceMessage


class OpenAIChatResponse(BaseModel):
    """Structured Pydantic response model for OpenAI Chat Completion API."""

    choices: list[OpenAIChatChoice]


T = TypeVar("T")


def _execute_with_retry(  # noqa: PLR0917
    action_fn: Callable[[], T],
    is_rate_limit_fn: Callable[[Exception], bool],
    provider_label: str,
    model_name: str,
    interaction_id: str,
    max_attempts: int = 10,
    initial_delay: float = 4.0,
    max_delay: float = 120.0,
) -> T:
    """Execute action_fn with exponential backoff and jitter on rate limits or API errors.

    Args:
        action_fn: Function executing the LLM request.
        is_rate_limit_fn: Predicate checking if an exception is due to rate limits (429).
        provider_label: Display name of provider for logging/errors.
        model_name: Name of the LLM model.
        interaction_id: Interaction logger transaction ID.
        max_attempts: Max retry iterations before raising RuntimeError.
        initial_delay: Initial retry delay in seconds.
        max_delay: Cap on exponential backoff delay in seconds.

    Returns:
        The result returned by action_fn.

    Raises:
        RuntimeError: If execution fails after max_attempts iterations.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return action_fn()
        except Exception as e:
            err_msg = str(e)
            is_rate_limit = is_rate_limit_fn(e)

            if attempt == max_attempts:
                log_response_finish(
                    interaction_id=interaction_id,
                    response=f"{provider_label} API call failed after {max_attempts} attempts: {err_msg}",
                    status="FAILED",
                    model_name=model_name,
                )
                raise RuntimeError(f"{provider_label} API call failed after {max_attempts} attempts: {err_msg}") from e

            delay = min(max_delay, initial_delay * (2.0 ** (attempt - 1))) + random.uniform(0, 2.0)
            total_delay = min(max_delay, delay)

            if is_rate_limit:
                print(
                    f"  ⚠️  Rate Limit (429) for {model_name}. "
                    f"Backing off for {total_delay:.1f}s (Attempt {attempt}/{max_attempts})...",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  ⚠️  {provider_label} API Error for {model_name} ({err_msg[:100]}). "
                    f"Retrying in {total_delay:.1f}s (Attempt {attempt}/{max_attempts})...",
                    file=sys.stderr,
                )
            time.sleep(total_delay)

    raise RuntimeError(f"{provider_label} API call failed after {max_attempts} attempts.")


@deprecated("Use PydanticAIRunner instead.")
class OpenAICompatibleRunner(BaseLLMRunner):
    """LLM Runner targeting OpenAI-compatible HTTP chat completion endpoints.

    Supports API key and GCP ADC Bearer token authentication strategies.
    """

    def __init__(self, model_name: str, base_url: str, auth_config: "LLMAuthConfig | None" = None) -> None:
        """Initialize OpenAICompatibleRunner.

        Args:
            model_name: Target model identifier.
            base_url: Base endpoint URL for the OpenAI-compatible service.
            auth_config: Authentication model instance (APIKeyAuthConfig or GCPADCAuthConfig).
        """
        self._model_name = model_name
        self.base_url = base_url
        self.auth_config = auth_config
        self._credentials: google.auth.credentials.Credentials | None = None
        self._auth_request: google.auth.transport.requests.Request | None = None

    @property
    def model_name(self) -> str:
        """Name of the target LLM model."""
        return self._model_name

    def _api_key(self) -> str | None:
        """Resolve API key or GCP ADC access token.

        Returns:
            Valid Bearer token / API key string, or None if unauthenticated (e.g. local LLM).

        Raises:
            ValueError: If authentication configuration is invalid or missing required keys.
        """
        if self.auth_config is None or isinstance(self.auth_config, NoAuthConfig):
            return None

        if isinstance(self.auth_config, GCPADCAuthConfig):
            if self._credentials is None:
                creds_obj, _ = google.auth.default()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                creds_typed = cast(google.auth.credentials.Credentials, creds_obj)
                scoped_creds = with_scopes_if_required(creds_typed, ["https://www.googleapis.com/auth/cloud-platform"])  # pyright: ignore[reportUnknownVariableType]
                self._credentials = cast(google.auth.credentials.Credentials, scoped_creds)
                self._auth_request = google.auth.transport.requests.Request()

            self._credentials.refresh(self._auth_request)  # type: ignore[no-untyped-call]
            token = self._credentials.token  # pyright: ignore[reportUnknownMemberType]
            if not isinstance(token, str) or not token:
                raise ValueError("Failed to obtain valid OAuth2 token using GCP ADC.")
            return token

        if isinstance(self.auth_config, APIKeyAuthConfig):
            if not self.auth_config.api_key:
                raise ValueError("API key must be provided when auth_type is API_KEY.")
            return self.auth_config.api_key

    def complete(self, prompt: str, system_instruction: str) -> str:
        """Execute chat completion request against OpenAI-compatible endpoint.

        Args:
            prompt: User prompt text.
            system_instruction: System instructions / role context.

        Returns:
            Completion text response.

        Raises:
            RuntimeError: If API call fails after max retries.
        """
        interaction_id = log_request_start(
            prompt=prompt,
            model_name=self._model_name,
            system_instruction=system_instruction,
            provider="openai-compatible",
        )
        api_key = self._api_key()
        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = OpenAIChatPayload(
            model=self._model_name,
            messages=[
                OpenAIChatMessage(role="system", content=system_instruction),
                OpenAIChatMessage(role="user", content=prompt),
            ],
            temperature=0.2,
        )
        base_url = self.base_url.rstrip("/")
        if "v1" not in base_url.lower() and "chat/completions" not in base_url.lower():
            base_url = f"{base_url}/v1"
        url = f"{base_url}/chat/completions"

        def _make_request() -> str:
            response = requests.post(url, json=payload.model_dump(), headers=headers, timeout=600)
            if response.status_code == 200:
                raw_data = response.json()
                validated_resp = OpenAIChatResponse.model_validate(raw_data)
                content = validated_resp.choices[0].message.content
                log_response_finish(
                    interaction_id=interaction_id,
                    response=content,
                    status="COMPLETED",
                    model_name=self._model_name,
                )
                return content

            err_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code == 429 or "rate limit" in response.text.lower():
                raise _RateLimitError(err_msg)
            raise RuntimeError(err_msg)

        def _is_rate_limit(e: Exception) -> bool:
            return isinstance(e, _RateLimitError) or "429" in str(e) or "rate limit" in str(e).lower()

        return _execute_with_retry(
            action_fn=_make_request,
            is_rate_limit_fn=_is_rate_limit,
            provider_label="OpenAI-compatible",
            model_name=self._model_name,
            interaction_id=interaction_id,
        )


@deprecated("Use PydanticAIRunner instead.")
class VertexAIRunner(BaseLLMRunner):
    """LLM Runner targeting Google Cloud Vertex AI Generative AI services."""

    def __init__(self, model_name: str, location: str | None = None) -> None:
        """Initialize VertexAIRunner.

        Args:
            model_name: Target Vertex AI model name (e.g. 'gemini-1.5-flash').
            location: GCP region location. Defaults to GCP_LOCATION env var or 'europe-west1'.

        Raises:
            ValueError: If GCP_PROJECT_ID environment variable is missing.
        """
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.location = location or os.environ.get("GCP_LOCATION", "europe-west1")
        if not self.project_id:
            raise ValueError(
                "GCP_PROJECT_ID environment variable is required for VertexAIRunner. "
                "Ensure it is set in your .env or system environment."
            )
        vertexai.init(project=self.project_id, location=self.location)
        self._model_name = model_name
        self.model = GenerativeModel(model_name)

    @property
    def model_name(self) -> str:
        """Name of the target LLM model."""
        return self._model_name

    def complete(self, prompt: str, system_instruction: str) -> str:
        """Execute text generation request using Vertex AI SDK.

        Args:
            prompt: User input prompt text.
            system_instruction: System instruction text defining role/context.

        Returns:
            Generated text content from Vertex AI model.

        Raises:
            RuntimeError: If generation fails after max retries.
        """
        interaction_id = log_request_start(
            prompt=prompt,
            model_name=self._model_name,
            system_instruction=system_instruction,
            provider="vertex",
        )
        vertexai.init(project=self.project_id, location=self.location)
        model = GenerativeModel(self._model_name, system_instruction=system_instruction)

        def _make_vertex_request() -> str:
            response = model.generate_content(prompt)
            if not response.text:
                raise RuntimeError("Empty Vertex AI response.")
            log_response_finish(
                interaction_id=interaction_id,
                response=response.text,
                status="COMPLETED",
                model_name=self._model_name,
            )
            return response.text

        def _is_vertex_rate_limit(e: Exception) -> bool:
            err_str = str(e)
            return (
                isinstance(e, (TooManyRequests, ResourceExhausted))
                or "429" in err_str
                or "Resource exhausted" in err_str
                or "Quota exceeded" in err_str
            )

        return _execute_with_retry(
            action_fn=_make_vertex_request,
            is_rate_limit_fn=_is_vertex_rate_limit,
            provider_label="Vertex AI",
            model_name=self._model_name,
            interaction_id=interaction_id,
            initial_delay=5.0,
        )


class MockRunner(BaseLLMRunner):
    """Offline Mock LLM Runner generating mock tax data and PII payloads for testing."""

    @property
    def model_name(self) -> str:
        """Name of the mock runner."""
        return "Mock"

    def complete(self, prompt: str, system_instruction: str) -> str:
        """Generate mock LLM response based on prompt context.

        Args:
            prompt: User prompt text.
            system_instruction: System prompt role instructions.

        Returns:
            JSON formatted mock response string.
        """
        if "personally identifiable information" in system_instruction.lower():
            return get_mock_pii_json()
        return get_mock_voter_json(prompt, system_instruction)
