"""UI configuration data models."""

from dataclasses import dataclass, field
from pathlib import Path

from backend.db_manager import DatabaseManager


@dataclass(frozen=True)
class UIConfig:
    """Centralized configuration passed to UI tabs and windows.

    Attributes:
        db: Database manager instance.
        sessions_base_dir: Base directory for storing session states.
    """

    db: DatabaseManager
    sessions_base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / ".sessions"
    )

    @property
    def chat_sessions_dir(self) -> Path:
        """Directory for general chat sessions."""
        return self.sessions_base_dir / "chat"

    @property
    def tax_filing_sessions_dir(self) -> Path:
        """Directory for Irish tax filing sessions."""
        return self.sessions_base_dir / "tax_filing"
