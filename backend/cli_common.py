"""Shared CLI argument parsing helpers and typed Pydantic models.

Provides standard path arguments and type-safe argument parsing for CLI scripts.
"""

from __future__ import annotations

import argparse
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from backend.config import AppConfig, load_config

T = TypeVar("T")


class CommonConfigArgs(BaseModel):
    """Base Pydantic model for CLI commands supporting standard path options."""

    model_config = ConfigDict(extra="ignore")

    data_dir: str | None = None
    db: str | None = None
    vector_db: str | None = None

    def resolve_app_config(self) -> AppConfig:
        """Resolve full AppConfig instance from parsed CLI arguments.

        Returns:
            AppConfig resolved according to the configuration hierarchy.
        """
        return load_config(
            cli_data_dir=self.data_dir,
            cli_db_path=self.db,
            cli_vector_db_path=self.vector_db,
        )


def build_common_config_parser() -> argparse.ArgumentParser:
    """Build a reusable parent ArgumentParser containing common path arguments.

    Returns:
        argparse.ArgumentParser with add_help=False suitable for use in `parents=[...]`.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_common_config_args(parser)
    return parser


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Register standard path arguments to an ArgumentParser or SubParser.

    Args:
        parser: ArgumentParser or SubParser instance to configure.
    """
    _ = parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to root data directory (defaults to TAX_DATA_DIR env or user config)",
    )
    _ = parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database file (defaults to TAX_DB_PATH env or user config)",
    )
    _ = parser.add_argument(
        "--vector-db",
        type=str,
        default=None,
        help="Path to vector SQLite database file (defaults to TAX_VECTOR_DB_PATH env or user config)",
    )


def parse_typed_args(
    parser: argparse.ArgumentParser,
    model_or_adapter: type[T] | TypeAdapter[T],
    args: list[str] | None = None,
) -> T:
    """Parse CLI arguments and validate into a strongly-typed Pydantic model.

    Args:
        parser: Configured ArgumentParser instance.
        model_or_adapter: Pydantic model class or TypeAdapter to validate against.
        args: Optional list of command-line argument strings (defaults to sys.argv[1:]).

    Returns:
        Validated strongly-typed model instance.
    """
    raw_ns = parser.parse_args(args)
    raw_dict: dict[str, object] = vars(raw_ns)

    if isinstance(model_or_adapter, TypeAdapter):
        return model_or_adapter.validate_python(raw_dict)
    if issubclass(model_or_adapter, BaseModel):
        return model_or_adapter.model_validate(raw_dict)

    adapter = TypeAdapter(model_or_adapter)
    return adapter.validate_python(raw_dict)
