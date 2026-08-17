"""Unit tests for centralized AppConfig management and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import (
    AppConfig,
    UserConfigFile,
    get_app_config,
    load_config,
    reset_app_config,
    save_user_config,
    set_app_config,
)


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given: No environment variables or custom config file set
    monkeypatch.delenv("TAX_DATA_DIR", raising=False)
    monkeypatch.delenv("TAX_DB_PATH", raising=False)
    monkeypatch.delenv("TAX_VECTOR_DB_PATH", raising=False)
    monkeypatch.setenv("TAX_CONFIG_PATH", str(tmp_path / "non_existent_config.json"))

    # When: Loading configuration without explicit overrides
    config = load_config(use_user_config=False)

    # Then: Default paths are resolved relative to CWD
    assert config.data_dir == (Path.cwd() / "data").resolve()
    assert config.db_path == (Path.cwd() / "database" / "tax_data.db").resolve()
    assert config.vector_db_path == (Path.cwd() / "database" / "tax_vectors.db").resolve()
    assert config.raw_sources_dir == (Path.cwd() / "data" / "raw_sources").resolve()
    assert config.research_dir == (Path.cwd() / "data" / "research").resolve()


def test_load_config_from_config_file(tmp_path: Path) -> None:
    # Given: A custom JSON configuration file adhering to UserConfigFile schema
    config_file = tmp_path / "custom_config.json"
    custom_data = tmp_path / "custom_data"
    custom_db = tmp_path / "custom_tax.db"
    custom_vector_db = tmp_path / "custom_vector.db"

    user_cfg = UserConfigFile(
        data_dir=str(custom_data),
        db_path=str(custom_db),
        vector_db_path=str(custom_vector_db),
    )
    config_file.write_text(user_cfg.model_dump_json(indent=2), encoding="utf-8")

    # When: Loading configuration using the custom file path
    config = load_config(config_file=config_file)

    # Then: Config file values are used
    assert config.data_dir == custom_data.resolve()
    assert config.db_path == custom_db.resolve()
    assert config.vector_db_path == custom_vector_db.resolve()


def test_load_config_fails_loud_on_invalid_json(tmp_path: Path) -> None:
    # Given: A corrupt / malformed JSON configuration file
    config_file = tmp_path / "corrupt_config.json"
    config_file.write_text("{ broken json ...", encoding="utf-8")

    # When/Then: Loading configuration raises ValueError explicitly instead of silently falling back
    with pytest.raises(ValueError, match="Failed to parse user configuration file"):
        load_config(config_file=config_file)


def test_load_config_fails_loud_on_schema_violation(tmp_path: Path) -> None:
    # Given: A JSON file with unexpected / forbidden extra keys
    config_file = tmp_path / "invalid_schema.json"
    config_file.write_text(
        json.dumps({"data_dir": "/path/data", "unknown_key": "illegal"}),
        encoding="utf-8",
    )

    # When/Then: Pydantic validation fails loud with ValueError
    with pytest.raises(ValueError, match="Failed to parse user configuration file"):
        load_config(config_file=config_file)


def test_load_config_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given: A config file, environment variables, and CLI overrides
    config_file = tmp_path / "config.json"
    user_cfg = UserConfigFile(
        data_dir=str(tmp_path / "file_data"),
        db_path=str(tmp_path / "file_db.db"),
        vector_db_path=str(tmp_path / "file_vec.db"),
    )
    config_file.write_text(user_cfg.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setenv("TAX_DATA_DIR", str(tmp_path / "env_data"))
    monkeypatch.setenv("TAX_DB_PATH", str(tmp_path / "env_db.db"))

    cli_override_data = tmp_path / "cli_data"

    # When: Loading config with explicit CLI override for data_dir
    config = load_config(
        cli_data_dir=cli_override_data,
        config_file=config_file,
    )

    # Then: CLI override beats ENV, ENV beats File, File beats Default
    assert config.data_dir == cli_override_data.resolve()
    assert config.db_path == (tmp_path / "env_db.db").resolve()
    assert config.vector_db_path == (tmp_path / "file_vec.db").resolve()


def test_save_user_config(tmp_path: Path) -> None:
    # Given: An isolated target config path and configuration values
    target_config = tmp_path / "saved_config.json"
    custom_data = tmp_path / "dataset"
    custom_db = tmp_path / "dbs" / "tax.db"
    custom_vec = tmp_path / "dbs" / "vec.db"

    app_config = AppConfig(
        data_dir=custom_data,
        db_path=custom_db,
        vector_db_path=custom_vec,
        sessions_base_dir=tmp_path / ".sessions",
    )

    # When: Saving configuration to disk
    saved_path = save_user_config(app_config, config_file=target_config)

    # Then: File is created and contains valid UserConfigFile JSON structure
    assert saved_path == target_config.resolve()
    assert target_config.is_file()

    loaded = UserConfigFile.model_validate_json(target_config.read_text(encoding="utf-8"))
    assert loaded.data_dir == str(custom_data)
    assert loaded.db_path == str(custom_db)
    assert loaded.vector_db_path == str(custom_vec)


def test_global_app_config_singleton(tmp_path: Path) -> None:
    # Given: A custom AppConfig instance
    reset_app_config()
    custom_config = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "tax.db",
        vector_db_path=tmp_path / "vec.db",
        sessions_base_dir=tmp_path / "sessions",
    )

    # When: Setting the global config
    set_app_config(custom_config)

    # Then: get_app_config() returns the configured instance
    assert get_app_config() == custom_config
    reset_app_config()
