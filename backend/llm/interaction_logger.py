"""Interaction logger for recording model requests and responses immediately to disk."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOG_LOCK = threading.Lock()
DEFAULT_LOG_DIR = Path("logs")
TEXT_LOG_FILENAME = "model_interactions.log"
JSONL_LOG_FILENAME = "model_interactions.jsonl"


def log_request_start(  # noqa: PLR0917
    prompt: str,
    model_name: str | None = None,
    system_instruction: str | None = None,
    provider: str | None = None,
    metadata: dict[str, Any] | None = None,
    log_dir: Path | str | None = None,
) -> str:
    """Log input request immediately to disk before model execution starts.

    Ensures that user prompts and system instructions are safely recorded
    even if an unexpected crash or exception occurs during model execution.

    Args:
        prompt: User input prompt or request text sent to the model.
        model_name: Optional name/identifier of the model.
        system_instruction: Optional system prompt or instruction text.
        provider: Optional provider name.
        metadata: Optional dictionary of additional context metrics.
        log_dir: Optional custom log directory path.

    Returns:
        Generated unique interaction ID string.
    """
    interaction_id = str(uuid.uuid4())[:8]
    target_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    model = model_name or "unknown"
    prov = provider or "unknown"

    text_log_path = target_dir / TEXT_LOG_FILENAME
    jsonl_log_path = target_dir / JSONL_LOG_FILENAME

    formatted_entry = (
        f"{'=' * 80}\n"
        f"TIMESTAMP: {timestamp_iso} | ID: {interaction_id} | STATUS: REQUEST_STARTED\n"
        f"MODEL: {model} | PROVIDER: {prov}\n"
        f"{'-' * 35} REQUEST {'-' * 36}\n"
    )
    if system_instruction:
        formatted_entry += f"[SYSTEM INSTRUCTION]\n{system_instruction}\n\n"
    formatted_entry += f"[PROMPT]\n{prompt}\n"
    formatted_entry += f"{'=' * 80}\n\n"

    json_record: dict[str, str | None | object] = {
        "interaction_id": interaction_id,
        "event_type": "request_start",
        "timestamp": timestamp_iso,
        "model_name": model,
        "provider": prov,
        "system_instruction": system_instruction,
        "prompt": prompt,
        "metadata": metadata or {},
    }

    try:
        with _LOG_LOCK:
            with text_log_path.open("a", encoding="utf-8") as f_text:
                f_text.write(formatted_entry)
                f_text.flush()

            with jsonl_log_path.open("a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                f_jsonl.flush()
    except Exception as exc:
        logger.error("Failed to write model request start log: %s", exc)

    return interaction_id


def log_response_finish(
    interaction_id: str,
    response: str,
    status: str = "COMPLETED",
    model_name: str | None = None,
    log_dir: Path | str | None = None,
) -> None:
    """Log completed or failed response to disk for a given interaction ID.

    Args:
        interaction_id: Unique interaction ID returned by log_request_start.
        response: Output text response or error string.
        status: Status indicator string ('COMPLETED', 'FAILED', 'CANCELLED').
        model_name: Optional model identifier.
        log_dir: Optional custom log directory path.
    """
    target_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    text_log_path = target_dir / TEXT_LOG_FILENAME
    jsonl_log_path = target_dir / JSONL_LOG_FILENAME

    formatted_entry = (
        f"{'-' * 80}\n"
        f"TIMESTAMP: {timestamp_iso} | ID: {interaction_id} | STATUS: RESPONSE_{status}\n"
        f"{'-' * 35} RESPONSE {'-' * 35}\n"
        f"{response}\n"
        f"{'=' * 80}\n\n"
    )

    json_record = {
        "interaction_id": interaction_id,
        "event_type": f"response_{status.lower()}",
        "timestamp": timestamp_iso,
        "model_name": model_name or "unknown",
        "status": status,
        "response": response,
    }

    try:
        with _LOG_LOCK:
            with text_log_path.open("a", encoding="utf-8") as f_text:
                f_text.write(formatted_entry)
                f_text.flush()

            with jsonl_log_path.open("a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                f_jsonl.flush()
    except Exception as exc:
        logger.error("Failed to write model response finish log: %s", exc)


def log_interaction(  # noqa: PLR0917
    prompt: str,
    response: str,
    model_name: str | None = None,
    system_instruction: str | None = None,
    provider: str | None = None,
    metadata: dict[str, Any] | None = None,
    log_dir: Path | str | None = None,
) -> None:
    """Log complete request and response pair atomically.

    Args:
        prompt: User input prompt text.
        response: Model output text.
        model_name: Optional model identifier.
        system_instruction: Optional system instruction text.
        provider: Optional provider name.
        metadata: Optional context metadata.
        log_dir: Optional custom log directory path.
    """
    interaction_id = log_request_start(
        prompt=prompt,
        model_name=model_name,
        system_instruction=system_instruction,
        provider=provider,
        metadata=metadata,
        log_dir=log_dir,
    )
    log_response_finish(
        interaction_id=interaction_id,
        response=response,
        status="COMPLETED",
        model_name=model_name,
        log_dir=log_dir,
    )
