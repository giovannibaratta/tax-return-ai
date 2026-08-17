"""Entry point for the Tax Return AI desktop application.

Run from the repository root with::

    python -m src.ui.main

Configuration is resolved via CLI arguments, environment variables,
the user config file (~/.config/tax-return-ai/config.json), or default paths.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet  # type: ignore[import-untyped]

from backend.cli_common import CommonConfigArgs, add_common_config_args, parse_typed_args
from backend.config import set_app_config
from backend.db_manager import DatabaseManager, LocalDb
from src.ui.config import UIConfig
from src.ui.main_window import MainWindow

_ = load_dotenv()


class UICliArgs(CommonConfigArgs):
    """CLI arguments for the desktop application entry point."""


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and configure ArgumentParser for the desktop application."""
    parser = argparse.ArgumentParser(description="Tax Return AI UI")
    add_common_config_args(parser)
    return parser


def main() -> None:
    """Parse CLI arguments, initialize configuration, apply theme, and open MainWindow."""
    parser = _build_arg_parser()
    args = parse_typed_args(parser, UICliArgs)
    app_config = args.resolve_app_config()
    set_app_config(app_config)

    # Create a single DatabaseManager shared across all tabs.
    db = DatabaseManager(
        db_config=LocalDb(
            db_path=app_config.db_path,
            vector_db_path=app_config.vector_db_path,
        )
    )
    config = UIConfig(db=db, app_config=app_config)

    app = QApplication(sys.argv)
    app.setApplicationName("Tax Return AI")

    apply_stylesheet(app, theme="light_blue.xml")

    window = MainWindow(config=config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
