"""Entry point for the Tax Return AI desktop application.

Run from the repository root with::

    python -m src.ui.main

The database path defaults to ``database/tax_data.db`` (relative to CWD) but
can be overridden via the ``TAX_DB_PATH`` environment variable or the ``--db``
CLI flag.
"""

import argparse
import sys

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet  # type: ignore[import-untyped]

from backend.db_manager import DEFAULT_DB_PATH, DatabaseManager, LocalDb
from src.ui.config import UIConfig
from src.ui.main_window import MainWindow

_ = load_dotenv()


def main() -> None:
    """Parse CLI arguments, run DB migrations once, apply theme, and open MainWindow."""
    parser = argparse.ArgumentParser(description="Tax Return AI UI")
    _ = parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (overrides TAX_DB_PATH env var)",
    )
    args = parser.parse_args()

    db_path: str = args.db or DEFAULT_DB_PATH

    # Create a single DatabaseManager shared across all tabs.
    db = DatabaseManager(db_config=LocalDb(db_path=db_path))
    config = UIConfig(db=db)

    app = QApplication(sys.argv)
    app.setApplicationName("Tax Return AI")

    apply_stylesheet(app, theme="light_blue.xml")

    window = MainWindow(config=config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
