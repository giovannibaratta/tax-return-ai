"""Unit tests for BaseLLMRunner factory configuration resolution and authentication."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.llm.auth_config import GCPADCAuthConfig, NoAuthConfig
from backend.llm.runner_factory import build_runner
from backend.llm.runners import OpenAICompatibleRunner, VertexAIRunner


def test_build_runner_openai_default() -> None:
    # Given: Default provider and model are set in environment
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "DEFAULT_LLM_BASE_URL": "http://localhost:9000",
        "DEFAULT_LLM_API_KEY": "some-key",
        "DEFAULT_LLM_AUTH_TYPE": "API_KEY",
    }
    with patch.dict(os.environ, env, clear=True):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns an OpenAICompatibleRunner using the defaults
        assert isinstance(runner, OpenAICompatibleRunner)
        assert runner.model_name == "gemma"
        assert runner.base_url == "http://localhost:9000"


def test_build_runner_openai_explicit() -> None:
    # Given: Slot-specific OpenAI provider and model settings overriding defaults
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "DEFAULT_LLM_BASE_URL": "http://localhost:9000",
        "DEFAULT_LLM_AUTH_TYPE": "API_KEY",
        "TEST_AGENT_PROVIDER": "openai-compatible",
        "TEST_AGENT_MODEL": "my-custom-model",
        "TEST_AGENT_BASE_URL": "http://my-api:8080",
        "TEST_AGENT_API_KEY": "my-key",
    }
    with patch.dict(os.environ, env, clear=True):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns an OpenAICompatibleRunner with slot-specific settings
        assert isinstance(runner, OpenAICompatibleRunner)
        assert runner.model_name == "my-custom-model"
        assert runner.base_url == "http://my-api:8080"


def test_build_runner_openai_auth_type_explicit() -> None:
    # Given: OpenAI-compatible provider with explicit GCP_ADC auth type
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "DEFAULT_LLM_BASE_URL": "http://localhost:9000",
        "TEST_AGENT_PROVIDER": "openai-compatible",
        "TEST_AGENT_MODEL": "my-custom-model",
        "TEST_AGENT_BASE_URL": "http://my-api:8080",
        "TEST_AGENT_AUTH_TYPE": "GCP_ADC",
    }
    with patch.dict(os.environ, env, clear=True):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns an OpenAICompatibleRunner configured with GCPADCAuthConfig
        assert isinstance(runner, OpenAICompatibleRunner)
        assert isinstance(runner.auth_config, GCPADCAuthConfig)


def test_build_runner_openai_auth_type_fallback() -> None:
    # Given: OpenAI-compatible provider with DEFAULT_LLM_AUTH_TYPE fallback
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "DEFAULT_LLM_BASE_URL": "http://localhost:9000",
        "TEST_AGENT_PROVIDER": "openai-compatible",
        "TEST_AGENT_MODEL": "my-custom-model",
        "TEST_AGENT_BASE_URL": "http://my-api:8080",
        "DEFAULT_LLM_AUTH_TYPE": "GCP_ADC",
    }
    with patch.dict(os.environ, env, clear=True):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns an OpenAICompatibleRunner with fallback GCPADCAuthConfig
        assert isinstance(runner, OpenAICompatibleRunner)
        assert isinstance(runner.auth_config, GCPADCAuthConfig)


def test_build_runner_openai_unauthenticated() -> None:
    # Given: OpenAI-compatible provider with auth_type set to NONE for local LLM
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "local-llama",
        "DEFAULT_LLM_BASE_URL": "http://localhost:11434",
        "TEST_AGENT_AUTH_TYPE": "NONE",
    }
    with patch.dict(os.environ, env, clear=True):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns an OpenAICompatibleRunner with NoAuthConfig
        assert isinstance(runner, OpenAICompatibleRunner)
        assert isinstance(runner.auth_config, NoAuthConfig)


def test_build_runner_vertex_explicit() -> None:
    # Given: Slot-specific Vertex provider, model name, location, and GCP project ID
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "TEST_AGENT_PROVIDER": "vertex",
        "TEST_AGENT_MODEL": "gemini-1.5-flash-002",
        "TEST_AGENT_LOCATION": "us-west1",
        "GCP_PROJECT_ID": "test-project-123",
    }
    with (
        patch("backend.llm.runners.vertexai") as mock_vertexai,
        patch("backend.llm.runners.GenerativeModel") as mock_gen_model,
        patch.dict(os.environ, env, clear=True),
    ):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns a VertexAIRunner initialized with slot-specific location
        assert isinstance(runner, VertexAIRunner)
        assert runner.location == "us-west1"
        assert runner.model_name == "gemini-1.5-flash-002"
        mock_vertexai.init.assert_called_once_with(project="test-project-123", location="us-west1")
        mock_gen_model.assert_called_once_with("gemini-1.5-flash-002")


def test_build_runner_vertex_fallback_model_and_location() -> None:
    # Given: Vertex provider with defaults for model and location
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemini-1.5-flash-002",
        "DEFAULT_GCP_LOCATION": "europe-west4",
        "TEST_AGENT_PROVIDER": "vertex",
        "GCP_PROJECT_ID": "test-project-123",
    }
    with (
        patch("backend.llm.runners.vertexai") as mock_vertexai,
        patch("backend.llm.runners.GenerativeModel") as mock_gen_model,
        patch.dict(os.environ, env, clear=True),
    ):
        # When: Building a runner with prefix "TEST_AGENT"
        runner = build_runner("TEST_AGENT")

        # Then: Returns a VertexAIRunner with default model and default location
        assert isinstance(runner, VertexAIRunner)
        assert runner.location == "europe-west4"
        assert runner.model_name == "gemini-1.5-flash-002"
        mock_vertexai.init.assert_called_once_with(project="test-project-123", location="europe-west4")
        mock_gen_model.assert_called_once_with("gemini-1.5-flash-002")


def test_build_runner_vertex_missing_project_id() -> None:
    # Given: Vertex provider configured but GCP_PROJECT_ID missing
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "TEST_AGENT_PROVIDER": "vertex",
        "TEST_AGENT_MODEL": "gemini-1.5-flash-002",
    }
    with (
        patch("backend.llm.runners.vertexai"),
        patch("backend.llm.runners.GenerativeModel"),
        patch.dict(os.environ, env, clear=True),
    ):
        # When / Then: Raises ValueError on missing GCP_PROJECT_ID
        with pytest.raises(ValueError) as excinfo:
            build_runner("TEST_AGENT")
        assert "GCP_PROJECT_ID environment variable is required" in str(excinfo.value)


def test_build_runner_unknown_provider() -> None:
    # Given: An unknown provider configured
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
        "TEST_AGENT_PROVIDER": "unsupported-provider",
    }
    with patch.dict(os.environ, env, clear=True):
        # When / Then: Raises ValueError for unknown provider
        with pytest.raises(ValueError) as excinfo:
            build_runner("TEST_AGENT")
        assert "Unknown provider 'unsupported-provider'" in str(excinfo.value)


def test_build_runner_missing_default_provider() -> None:
    # Given: DEFAULT_LLM_PROVIDER missing in environment
    with patch.dict(os.environ, {}, clear=True):
        # When / Then: Raises ValueError
        with pytest.raises(ValueError) as excinfo:
            build_runner("TEST_AGENT")
        assert "DEFAULT_LLM_PROVIDER environment variable is not set" in str(excinfo.value)


def test_build_runner_missing_default_model() -> None:
    # Given: DEFAULT_LLM_MODEL missing in environment
    env = {"DEFAULT_LLM_PROVIDER": "openai-compatible"}
    with patch.dict(os.environ, env, clear=True):
        # When / Then: Raises ValueError
        with pytest.raises(ValueError) as excinfo:
            build_runner("TEST_AGENT")
        assert "DEFAULT_LLM_MODEL environment variable is not set" in str(excinfo.value)


def test_build_runner_missing_base_url_for_openai() -> None:
    # Given: DEFAULT_LLM_BASE_URL and prefix BASE_URL missing for openai-compatible
    env = {
        "DEFAULT_LLM_PROVIDER": "openai-compatible",
        "DEFAULT_LLM_MODEL": "gemma",
    }
    with patch.dict(os.environ, env, clear=True):
        # When / Then: Raises ValueError
        with pytest.raises(ValueError) as excinfo:
            build_runner("TEST_AGENT")
        assert "are both not set" in str(excinfo.value)


def test_openai_compatible_runner_gcp_adc_token_fetching() -> None:
    # Given: OpenAICompatibleRunner configured with auth_type GCP_ADC
    runner = OpenAICompatibleRunner(
        model_name="my-model",
        base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/endpoints/openapi",
        auth_config=GCPADCAuthConfig(),
    )

    with (
        patch("google.auth.default") as mock_auth_default,
        patch("google.auth.transport.requests.Request") as _,
        patch("requests.post") as mock_post,
    ):
        mock_creds = MagicMock()
        mock_creds.token = "mock-gcp-token"
        mock_auth_default.return_value = (mock_creds, "mock-project")

        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "response content"}}]}

        # When: Executing completion
        result = runner.complete("hello", "you are helper")

        # Then: Returns completion content and injects Bearer token
        assert result == "response content"
        mock_auth_default.assert_called_once()
        mock_creds.refresh.assert_called_once()

        called_args, called_kwargs = mock_post.call_args
        assert called_kwargs["headers"]["Authorization"] == "Bearer mock-gcp-token"
