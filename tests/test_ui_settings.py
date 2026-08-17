"""Unit tests for UI Settings tab and configuration reloading."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from backend.config import AppConfig
from backend.db_manager import DatabaseManager, MemoryDb
from src.ui.config import UIConfig
from src.ui.settings_tab import SettingsTab


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


def test_settings_tab_init_and_apply(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: An isolated environment, test AppConfig and DatabaseManager
    test_cfg_file = tmp_path / "user_config.json"
    monkeypatch.setenv("TAX_CONFIG_PATH", str(test_cfg_file))

    app_config = AppConfig(
        data_dir=tmp_path / "initial_data",
        db_path=tmp_path / "initial.db",
        vector_db_path=tmp_path / "initial_vec.db",
        sessions_base_dir=tmp_path / "sessions",
    )
    db = DatabaseManager(MemoryDb())
    ui_config = UIConfig(db=db, app_config=app_config)

    # When: Initializing SettingsTab
    tab = SettingsTab(config=ui_config)

    # Then: Fields match initial configuration
    assert tab._txt_data_dir.text() == str(app_config.data_dir)
    assert tab._txt_db_path.text() == str(app_config.db_path)
    assert tab._txt_vector_db_path.text() == str(app_config.vector_db_path)

    # When: Updating fields and saving
    new_data_dir = tmp_path / "updated_data"
    new_db_path = tmp_path / "updated.db"
    tab._txt_data_dir.setText(str(new_data_dir))
    tab._txt_db_path.setText(str(new_db_path))

    emitted_configs: list[UIConfig] = []

    def on_config_updated(cfg: UIConfig) -> None:
        emitted_configs.append(cfg)

    tab.config_updated.connect(on_config_updated)

    tab._apply_hot_reload()

    # Then: Signal is emitted with new values
    assert len(emitted_configs) == 1
    updated_cfg = emitted_configs[0]
    assert updated_cfg.data_dir == new_data_dir.resolve()
    assert updated_cfg.db_path == new_db_path.resolve()
