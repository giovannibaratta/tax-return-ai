"""Centralized application configuration management.

Provides a unified AppConfig model resolving paths for:
1. Data directory (raw_sources, research, scenarios, processed)
2. Relational SQLite database
3. Vector SQLite database
4. Chat and filing session storage

Resolution hierarchy:
1. Explicit CLI / function arguments
2. Environment variables (`TAX_DATA_DIR`, `TAX_DB_PATH`, `TAX_VECTOR_DB_PATH`)
3. User config file (`~/.config/tax-return-ai/config.json` or `TAX_CONFIG_PATH`)
4. Explicit default relative paths
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATA_DIR_NAME = "data"
DEFAULT_DB_PATH_NAME = "database/tax_data.db"
DEFAULT_VECTOR_DB_PATH_NAME = "database/tax_vectors.db"
DEFAULT_SESSIONS_DIR_NAME = ".sessions"


class UserConfigFile(BaseModel):
    """Schema for persistent user configuration JSON file (~/.config/tax-return-ai/config.json)."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str | None = Field(default=None, description="Path to dataset directory")
    db_path: str | None = Field(default=None, description="Path to relational SQLite database")
    vector_db_path: str | None = Field(default=None, description="Path to vector SQLite database")


def get_user_config_path() -> Path:
    """Return path to user-level configuration JSON file.

    Checks ``TAX_CONFIG_PATH`` environment variable first. If unset,
    defaults to ``~/.config/tax-return-ai/config.json``.

    Returns:
        Path pointing to configuration JSON file.
    """
    env_cfg = os.environ.get("TAX_CONFIG_PATH")
    if env_cfg:
        return Path(env_cfg).resolve()
    return Path.home() / ".config" / "tax-return-ai" / "config.json"


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration containing resolved paths.

    Attributes:
        data_dir: Root directory holding dataset folders.
        db_path: Path to relational SQLite database.
        vector_db_path: Path to vector SQLite database.
        sessions_base_dir: Base directory for storing session states.
    """

    data_dir: Path
    db_path: Path
    vector_db_path: Path
    sessions_base_dir: Path

    @property
    def raw_sources_dir(self) -> Path:
        """Directory for raw regulation and source documents."""
        return self.data_dir / "raw_sources"

    @property
    def raw_regulations_dir(self) -> Path:
        """Directory for raw regulation PDFs."""
        return self.data_dir / "raw_sources" / "regulations"

    @property
    def raw_records_dir(self) -> Path:
        """Directory for raw transaction records (PDF/CSV)."""
        return self.data_dir / "raw_sources" / "records"

    @property
    def research_dir(self) -> Path:
        """Directory for research markdown documents."""
        return self.data_dir / "research"

    @property
    def scenarios_dir(self) -> Path:
        """Directory for taxpayer scenarios."""
        return self.data_dir / "scenarios"

    @property
    def processed_dir(self) -> Path:
        """Directory for generated outputs and transcripts."""
        return self.data_dir / "processed"


_GLOBAL_APP_CONFIG: AppConfig | None = None


def _load_user_config_file(
    config_file: str | Path | None = None,
    use_user_config: bool = True,
) -> UserConfigFile | None:
    """Read and validate user config JSON file if enabled and present on disk.

    Args:
        config_file: Optional explicit path to configuration file.
        use_user_config: Whether to attempt reading user configuration.

    Returns:
        Validated UserConfigFile instance, or None if disabled or file absent.

    Raises:
        ValueError: If configuration file exists but fails validation.
    """
    if not use_user_config:
        return None

    cfg_path = Path(config_file).resolve() if config_file else get_user_config_path()
    if not cfg_path.is_file():
        return None

    try:
        content = cfg_path.read_text(encoding="utf-8")
        return UserConfigFile.model_validate_json(content)
    except Exception as e:
        raise ValueError(
            f"Failed to parse user configuration file at '{cfg_path}': {e}"
        ) from e


def _resolve_path(
    cli_override: str | Path | None,
    env_var_name: str,
    file_override: str | None,
    default_name: str,
) -> Path:
    """Resolve a single path following the precedence hierarchy.

    Args:
        cli_override: Optional explicit CLI / function argument.
        env_var_name: Name of environment variable to check.
        file_override: Optional path string from user config file.
        default_name: Default relative path from current working directory.

    Returns:
        Resolved absolute Path.
    """
    if cli_override is not None:
        return Path(cli_override).resolve()

    env_val = os.environ.get(env_var_name, "").strip()
    if env_val:
        return Path(env_val).resolve()

    if file_override and file_override.strip():
        return Path(file_override.strip()).resolve()

    return (Path.cwd() / default_name).resolve()


def load_config(
    cli_data_dir: str | Path | None = None,
    cli_db_path: str | Path | None = None,
    cli_vector_db_path: str | Path | None = None,
    config_file: str | Path | None = None,
    use_user_config: bool = True,
) -> AppConfig:
    """Resolve and return an AppConfig instance following the configuration hierarchy.

    Precedence:
    1. Explicit CLI / function arguments
    2. Environment variables (TAX_DATA_DIR, TAX_DB_PATH, TAX_VECTOR_DB_PATH)
    3. User config JSON file (~/.config/tax-return-ai/config.json or TAX_CONFIG_PATH)
    4. Default paths

    Args:
        cli_data_dir: Optional explicit data directory override.
        cli_db_path: Optional explicit relational DB path override.
        cli_vector_db_path: Optional explicit vector DB path override.
        config_file: Optional path to config file (overrides default user config path).
        use_user_config: If False, skip reading from user configuration file on disk.

    Returns:
        Fully resolved AppConfig instance.

    Raises:
        ValueError: If user configuration JSON is invalid or fails schema validation.
    """
    file_cfg = _load_user_config_file(config_file, use_user_config)

    resolved_data_dir = _resolve_path(
        cli_override=cli_data_dir,
        env_var_name="TAX_DATA_DIR",
        file_override=file_cfg.data_dir if file_cfg else None,
        default_name=DEFAULT_DATA_DIR_NAME,
    )
    resolved_db_path = _resolve_path(
        cli_override=cli_db_path,
        env_var_name="TAX_DB_PATH",
        file_override=file_cfg.db_path if file_cfg else None,
        default_name=DEFAULT_DB_PATH_NAME,
    )
    resolved_vector_db_path = _resolve_path(
        cli_override=cli_vector_db_path,
        env_var_name="TAX_VECTOR_DB_PATH",
        file_override=file_cfg.vector_db_path if file_cfg else None,
        default_name=DEFAULT_VECTOR_DB_PATH_NAME,
    )
    resolved_sessions_dir = (Path.cwd() / DEFAULT_SESSIONS_DIR_NAME).resolve()

    return AppConfig(
        data_dir=resolved_data_dir,
        db_path=resolved_db_path,
        vector_db_path=resolved_vector_db_path,
        sessions_base_dir=resolved_sessions_dir,
    )


def _build_user_config_to_save(
    config_values: UserConfigFile | AppConfig | dict[str, str | None],
    target_path: Path,
) -> UserConfigFile:
    """Construct a validated UserConfigFile object from diverse input types.

    Args:
        config_values: Configuration source object.
        target_path: Target path to existing configuration file if merging.

    Returns:
        Validated UserConfigFile instance.

    Raises:
        ValueError: If configuration values violate schema.
    """
    if isinstance(config_values, UserConfigFile):
        return config_values
    if isinstance(config_values, AppConfig):
        return UserConfigFile(
            data_dir=str(config_values.data_dir),
            db_path=str(config_values.db_path),
            vector_db_path=str(config_values.vector_db_path),
        )

    existing: dict[str, str | None] = {}
    if target_path.is_file():
        try:
            content = target_path.read_text(encoding="utf-8")
            existing = UserConfigFile.model_validate_json(content).model_dump()
        except Exception as e:
            raise ValueError(
                f"Failed to parse existing configuration file at '{target_path}': {e}"
            ) from e
    existing.update(config_values)
    return UserConfigFile.model_validate(existing)


def save_user_config(
    config_values: UserConfigFile | AppConfig | dict[str, str | None],
    config_file: str | Path | None = None,
) -> Path:
    """Save configuration values into the user configuration JSON file after validation.

    Args:
        config_values: UserConfigFile, AppConfig, or dict of overrides.
        config_file: Optional target config file path. Defaults to get_user_config_path().

    Returns:
        Path to the saved configuration file.

    Raises:
        ValueError: If config values break the UserConfigFile schema.
    """
    target_path = Path(config_file).resolve() if config_file else get_user_config_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    user_cfg = _build_user_config_to_save(config_values, target_path)
    target_path.write_text(user_cfg.model_dump_json(indent=2), encoding="utf-8")
    return target_path


def get_app_config() -> AppConfig:
    """Retrieve global AppConfig instance, lazily resolving if not initialized.

    Returns:
        Current global AppConfig.
    """
    global _GLOBAL_APP_CONFIG
    if _GLOBAL_APP_CONFIG is None:
        _GLOBAL_APP_CONFIG = load_config()
    return _GLOBAL_APP_CONFIG


def set_app_config(config: AppConfig) -> None:
    """Set the global AppConfig instance.

    Args:
        config: New AppConfig instance to set globally.
    """
    global _GLOBAL_APP_CONFIG
    _GLOBAL_APP_CONFIG = config


def reset_app_config() -> None:
    """Reset the global AppConfig instance back to uninitialized state."""
    global _GLOBAL_APP_CONFIG
    _GLOBAL_APP_CONFIG = None
