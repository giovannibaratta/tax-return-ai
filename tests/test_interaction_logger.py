"""Tests for model interaction logger module."""

import json
from pathlib import Path

from backend.llm.interaction_logger import (
    log_interaction,
    log_request_start,
    log_response_finish,
)


def test_log_interaction_creates_formatted_text_and_jsonl(tmp_path: Path):
    # Given: Target temporary log directory and sample request/response payload
    log_dir = tmp_path / "logs"
    prompt_text = "What is the tax rate on capital gains in Italy?"
    response_text = "The substitute tax rate on financial capital gains in Italy is 26%."
    model_name = "test-model-v1"
    system_inst = "You are a strict tax law assistant."
    provider = "openai-compatible"

    # When: log_interaction is invoked with request and response details
    log_interaction(
        prompt=prompt_text,
        response=response_text,
        model_name=model_name,
        system_instruction=system_inst,
        provider=provider,
        log_dir=log_dir,
    )

    # Then: Both text log and JSONL log files are created in the target directory
    text_log_file = log_dir / "model_interactions.log"
    jsonl_log_file = log_dir / "model_interactions.jsonl"

    assert text_log_file.exists()
    assert jsonl_log_file.exists()

    # Then: Text log contains ISO timestamp header, prompt, and response
    text_content = text_log_file.read_text(encoding="utf-8")
    assert "TIMESTAMP:" in text_content
    assert "MODEL: test-model-v1" in text_content
    assert "PROVIDER: openai-compatible" in text_content
    assert prompt_text in text_content
    assert response_text in text_content
    assert system_inst in text_content

    # Then: JSONL file contains valid JSON line with structured attributes
    lines = jsonl_log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    record = json.loads(lines[0])
    assert record["model_name"] == model_name
    assert record["provider"] == provider
    assert record["prompt"] == prompt_text
    assert "timestamp" in record


def test_immediate_request_start_logging(tmp_path: Path):
    # Given: Target log dir and prompt
    log_dir = tmp_path / "logs"
    prompt_text = "Detailed tax query"

    # When: log_request_start is called at start of interaction
    interaction_id = log_request_start(
        prompt=prompt_text,
        model_name="immediate-model",
        system_instruction="System prompt text",
        provider="test-provider",
        log_dir=log_dir,
    )

    # Then: Log files immediately exist and contain REQUEST_STARTED before response is logged
    text_log_file = log_dir / "model_interactions.log"
    assert text_log_file.exists()
    content = text_log_file.read_text(encoding="utf-8")
    assert "STATUS: REQUEST_STARTED" in content
    assert interaction_id in content
    assert prompt_text in content

    # When: log_response_finish is called after completion
    log_response_finish(
        interaction_id=interaction_id,
        response="Completed response text",
        status="COMPLETED",
        model_name="immediate-model",
        log_dir=log_dir,
    )

    # Then: Log file contains RESPONSE_COMPLETED section
    updated_content = text_log_file.read_text(encoding="utf-8")
    assert "STATUS: RESPONSE_COMPLETED" in updated_content
    assert "Completed response text" in updated_content
