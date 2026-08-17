"""UI configuration data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.config import AppConfig, get_app_config
from backend.db_manager import DatabaseManager, LocalDb


@dataclass(frozen=True)
class UIConfig:
    """Centralized configuration passed to UI tabs and windows.

    Attributes:
        db: Database manager instance.
        app_config: Underlying AppConfig instance with resolved paths.
    """

    db: DatabaseManager
    app_config: AppConfig = field(default_factory=get_app_config)

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return self.app_config.data_dir

    @property
    def db_path(self) -> Path:
        """Relational database path."""
        return self.app_config.db_path

    @property
    def vector_db_path(self) -> Path:
        """Vector database path."""
        return self.app_config.vector_db_path

    @property
    def sessions_base_dir(self) -> Path:
        """Base directory for storing session states."""
        return self.app_config.sessions_base_dir

    @property
    def chat_sessions_dir(self) -> Path:
        """Directory for general chat sessions."""
        return self.app_config.sessions_base_dir / "chat"

    @property
    def tax_filing_sessions_dir(self) -> Path:
        """Directory for Irish tax filing sessions."""
        return self.app_config.sessions_base_dir / "tax_filing"

    def with_app_config(
        self,
        new_app_config: AppConfig,
        new_db: DatabaseManager | None = None,
    ) -> UIConfig:
        """Return a new UIConfig instance with updated AppConfig and DatabaseManager.

        Args:
            new_app_config: Updated AppConfig instance.
            new_db: Optional pre-created DatabaseManager; if None, creates new LocalDb.

        Returns:
            New UIConfig instance.
        """
        if new_db is None:
            new_db = DatabaseManager(
                db_config=LocalDb(
                    db_path=new_app_config.db_path,
                    vector_db_path=new_app_config.vector_db_path,
                )
            )
        return UIConfig(db=new_db, app_config=new_app_config)
