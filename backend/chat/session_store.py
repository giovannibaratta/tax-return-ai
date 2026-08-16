"""Local storage manager for chat sessions stored as JSON files in .sessions subdirectories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generic, TypeVar

from backend.chat.models import ChatSession

logger = logging.getLogger(__name__)

SessionT = TypeVar("SessionT", bound=ChatSession)


class SessionStore(Generic[SessionT]):
    """Manages persistence of chat sessions to disk in `.sessions` subdirectories.

    Attributes:
        sessions_dir: Path to directory storing session JSON files.
        session_cls: Type of ChatSession model for deserialization.
    """

    def __init__(
        self,
        sessions_dir: str | Path | None = None,
        session_cls: type[SessionT] = ChatSession,
    ) -> None:
        """Initialize session store directory.

        Args:
            sessions_dir: Optional path to sessions directory.
                Defaults to '.sessions/chat' in project root.
            session_cls: Model class used to validate/deserialize session JSON files.
        """
        if sessions_dir is None:
            # Default to project root .sessions/chat directory
            project_root = Path(__file__).resolve().parents[2]
            sessions_dir = project_root / ".sessions" / "chat"

        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_cls = session_cls

    def _file_path(self, session_id: str) -> Path:
        """Return file path for session ID."""
        return self.sessions_dir / f"{session_id}.json"

    def list_sessions(self) -> list[SessionT]:
        """List all chat sessions sorted by updated_at descending.

        Returns:
            List of session objects without corrupt or unparseable files.
        """
        sessions: list[SessionT] = []
        for file_path in self.sessions_dir.glob("*.json"):
            try:
                content = file_path.read_text(encoding="utf-8")
                session = self.session_cls.model_validate_json(content)
                sessions.append(session)
            except Exception as err:
                logger.warning("Failed to parse session file %s: %s", file_path, err)

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def load_session(self, session_id: str) -> SessionT | None:
        """Load a single session by session ID.

        Args:
            session_id: Unique session identifier.

        Returns:
            Loaded session or None if not found/invalid.
        """
        file_path = self._file_path(session_id)
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8")
            return self.session_cls.model_validate_json(content)
        except Exception as err:
            logger.error("Failed to read session %s: %s", session_id, err)
            return None

    def save_session(self, session: SessionT) -> None:
        """Save or overwrite session to disk.

        Args:
            session: Session instance to save.
        """
        file_path = self._file_path(session.id)
        json_data = session.model_dump_json(indent=2)
        _ = file_path.write_text(json_data, encoding="utf-8")
        logger.info("Saved session %s to %s", session.id, file_path)

    def delete_session(self, session_id: str) -> bool:
        """Delete session file from disk.

        Args:
            session_id: Unique identifier of session to remove.

        Returns:
            True if session file existed and was deleted, False otherwise.
        """
        file_path = self._file_path(session_id)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted session %s", session_id)
            return True
        return False

    def create_session(self, title: str = "New Chat", auto_save: bool = False, **kwargs: object) -> SessionT:
        """Create a new chat session.

        Args:
            title: Title of new chat session.
            auto_save: If True, immediately writes session to disk. Defaults to False.
            **kwargs: Additional fields for specialized ChatSession models.

        Returns:
            Newly created session instance.
        """
        session_data: dict[str, object] = {"title": title, **kwargs}
        session = self.session_cls.model_validate(session_data)
        if auto_save:
            self.save_session(session)
        return session
