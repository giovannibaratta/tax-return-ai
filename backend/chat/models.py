"""Data models for local chat sessions, messages, and tool call traces."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field

from backend.utils.agents import ToolCallInfo


class AttachedContextDoc(BaseModel):
    """Document attached as context from data folder.

    Attributes:
        relative_path: Path relative to data directory or repository root.
        full_path: Absolute path to document on disk.
        content: Raw text content of document.
        char_count: Character count of document content.
    """

    relative_path: str
    full_path: str
    content: str
    char_count: int


class BaseChatMessage(BaseModel):
    """Base class for all chat message variants."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TokenUsageInfo(BaseModel):
    """LLM execution token usage metrics.

    Attributes:
        request_tokens: Prompt input tokens.
        response_tokens: Model output tokens.
        total_tokens: Total turn tokens.
        cached_tokens: Cached input tokens if reported by provider.
        requests: Number of LLM requests / tool turns.
    """

    request_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    requests: int = 1


class UserChatMessage(BaseChatMessage):
    """User message turn (cannot contain tool calls)."""

    role: Literal["user"] = "user"


class AssistantChatMessage(BaseChatMessage):
    """Assistant message turn (may contain tool call traces and token usage)."""

    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCallInfo] = Field(default_factory=lambda: list[ToolCallInfo]())
    usage: TokenUsageInfo | None = None


class SystemChatMessage(BaseChatMessage):
    """System message turn (instructions / prompts)."""

    role: Literal["system"] = "system"


# Discriminated union for type-safe message parsing and tool call restriction
ChatMessage = Annotated[
    UserChatMessage | AssistantChatMessage | SystemChatMessage,
    Discriminator("role"),
]


class ChatSession(BaseModel):
    """Complete chat session including full message history.

    Attributes:
        id: Unique identifier for chat session.
        title: User-facing title for session.
        created_at: ISO format UTC timestamp when session created.
        updated_at: ISO format UTC timestamp when last message added.
        messages: Chronological sequence of messages in session.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    messages: list[ChatMessage] = Field(default_factory=lambda: list[ChatMessage]())
