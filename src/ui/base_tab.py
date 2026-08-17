"""Base widget class for application tabs."""

from __future__ import annotations

from abc import abstractmethod

from PySide6.QtWidgets import QWidget

from src.ui.config import UIConfig


class BaseAppTab(QWidget):
    """Base class for UI tabs supporting configuration reload and lifecycle management."""

    @abstractmethod
    def reload_config(self, config: UIConfig) -> None:
        """Reload state, dependencies, and database connections from new UIConfig.

        Args:
            config: Newly applied UIConfig instance.
        """
        raise NotImplementedError
