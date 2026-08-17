"""PySide6 Settings Tab for managing application paths and database connections."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from backend.config import (
    AppConfig,
    get_user_config_path,
    load_config,
    save_user_config,
    set_app_config,
)
from src.ui.base_tab import BaseAppTab
from src.ui.config import UIConfig


class SettingsTab(BaseAppTab):
    """Configuration management tab for viewing and updating system paths.

    Signals:
        config_updated: Emitted when settings are applied or saved, passing new UIConfig.
    """

    config_updated: Signal = Signal(UIConfig)

    def __init__(self, config: UIConfig, parent: QWidget | None = None) -> None:
        """Initialize SettingsTab.

        Args:
            config: Current UIConfig instance.
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self._config = config
        self._init_ui()
        self._load_from_config(self._config.app_config)

    def _init_ui(self) -> None:
        """Construct the Settings UI layout."""
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QLabel("⚙️ Application Settings & Paths")
        header_font = header.font()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        user_cfg_path = get_user_config_path()
        cfg_info = QLabel(
            f"Configure system dataset paths and database locations. Settings saved here are persisted to:\n"
            f"<b>{user_cfg_path}</b>"
        )
        cfg_info.setWordWrap(True)
        cfg_info.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(cfg_info)

        # 1. Data Directory Group
        data_group = QGroupBox("📁 Dataset Directory")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(8)

        data_desc = QLabel(
            "Root directory containing dataset folders (raw_sources/, research/, scenarios/, processed/)."
        )
        data_desc.setWordWrap(True)
        data_desc.setStyleSheet("color: #475569; font-size: 11px;")
        data_layout.addWidget(data_desc)

        data_row = QHBoxLayout()
        self._txt_data_dir = QLineEdit()
        self._txt_data_dir.setPlaceholderText("/path/to/tax-return-ai-dataset/data")
        self._txt_data_dir.textChanged.connect(self._on_paths_changed)
        data_row.addWidget(self._txt_data_dir)

        btn_browse_data = QPushButton("Browse...")
        btn_browse_data.clicked.connect(self._browse_data_dir)
        data_row.addWidget(btn_browse_data)
        data_layout.addLayout(data_row)

        self._lbl_data_status = QLabel()
        self._lbl_data_status.setWordWrap(True)
        self._lbl_data_status.setStyleSheet("font-size: 11px; padding: 4px;")
        data_layout.addWidget(self._lbl_data_status)

        layout.addWidget(data_group)

        # 2. Database Paths Group
        db_group = QGroupBox("🗄️ Database Locations")
        db_layout = QVBoxLayout(db_group)
        db_layout.setSpacing(10)

        # Relational DB
        lbl_rel = QLabel("<b>Relational Database (tax_data.db):</b>")
        db_layout.addWidget(lbl_rel)
        rel_row = QHBoxLayout()
        self._txt_db_path = QLineEdit()
        self._txt_db_path.setPlaceholderText("/path/to/database/tax_data.db")
        self._txt_db_path.textChanged.connect(self._on_paths_changed)
        rel_row.addWidget(self._txt_db_path)
        btn_browse_db = QPushButton("Browse...")
        btn_browse_db.clicked.connect(self._browse_db_path)
        rel_row.addWidget(btn_browse_db)
        db_layout.addLayout(rel_row)

        # Vector DB
        lbl_vec = QLabel("<b>Vector Database (tax_vectors.db):</b>")
        db_layout.addWidget(lbl_vec)
        vec_row = QHBoxLayout()
        self._txt_vector_db_path = QLineEdit()
        self._txt_vector_db_path.setPlaceholderText("/path/to/database/tax_vectors.db")
        self._txt_vector_db_path.textChanged.connect(self._on_paths_changed)
        vec_row.addWidget(self._txt_vector_db_path)
        btn_browse_vec = QPushButton("Browse...")
        btn_browse_vec.clicked.connect(self._browse_vector_db_path)
        vec_row.addWidget(btn_browse_vec)
        db_layout.addLayout(vec_row)

        layout.addWidget(db_group)

        # 3. Actions Row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(10)

        self._btn_save = QPushButton("💾 Save to User Config")
        self._btn_save.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; padding: 8px 16px;"
        )
        self._btn_save.clicked.connect(self._save_and_apply)
        actions_row.addWidget(self._btn_save)

        self._btn_apply = QPushButton("🔄 Apply & Reload")
        self._btn_apply.setStyleSheet("padding: 8px 16px;")
        self._btn_apply.clicked.connect(self._apply_hot_reload)
        actions_row.addWidget(self._btn_apply)

        btn_reset = QPushButton("↺ Reset to Defaults")
        btn_reset.setStyleSheet("padding: 8px 16px;")
        btn_reset.clicked.connect(self._reset_defaults)
        actions_row.addWidget(btn_reset)

        actions_row.addStretch()
        layout.addLayout(actions_row)

        self._lbl_action_status = QLabel()
        self._lbl_action_status.setStyleSheet("font-weight: bold; color: #16a34a;")
        layout.addWidget(self._lbl_action_status)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _load_from_config(self, app_config: AppConfig) -> None:
        """Populate line edits from an AppConfig instance.

        Args:
            app_config: AppConfig instance to read from.
        """
        self._txt_data_dir.setText(str(app_config.data_dir))
        self._txt_db_path.setText(str(app_config.db_path))
        self._txt_vector_db_path.setText(str(app_config.vector_db_path))
        self._update_data_status(app_config.data_dir)

    def _on_paths_changed(self) -> None:
        """Validate and update status indicators as user edits text fields."""
        path_str = self._txt_data_dir.text().strip()
        if path_str:
            self._update_data_status(Path(path_str))
        else:
            self._lbl_data_status.setText("<span style='color: #dc2626;'>⚠️ Data directory path is empty.</span>")

    def _update_data_status(self, data_path: Path) -> None:
        """Inspect data directory and report folder health status.

        Args:
            data_path: Path to inspect.
        """
        if not data_path.exists():
            self._lbl_data_status.setText(
                f"<span style='color: #dc2626;'>❌ Directory not found on disk: {data_path}</span>"
            )
            return

        if not data_path.is_dir():
            self._lbl_data_status.setText(
                f"<span style='color: #dc2626;'>❌ Path is not a directory: {data_path}</span>"
            )
            return

        subdirs: list[str] = []
        for name in ("raw_sources", "research", "scenarios", "processed"):
            p = data_path / name
            if p.is_dir():
                subdirs.append(f"<span style='color: #16a34a;'>✅ {name}/</span>")
            else:
                subdirs.append(f"<span style='color: #94a3b8;'>⚠️ {name}/</span>")

        self._lbl_data_status.setText("Status: " + "  |  ".join(subdirs))

    def _browse_data_dir(self) -> None:
        """Open directory chooser for data folder."""
        current = self._txt_data_dir.text().strip() or str(Path.cwd())
        selected = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", current)
        if selected:
            self._txt_data_dir.setText(selected)

    def _browse_db_path(self) -> None:
        """Open file chooser for relational DB."""
        current = self._txt_db_path.text().strip() or str(Path.cwd() / "database")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select Relational Database File",
            current,
            "SQLite Database (*.db *.sqlite *.sqlite3);;All Files (*)",
        )
        if selected:
            self._txt_db_path.setText(selected)

    def _browse_vector_db_path(self) -> None:
        """Open file chooser for vector DB."""
        current = self._txt_vector_db_path.text().strip() or str(Path.cwd() / "database")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select Vector Database File",
            current,
            "SQLite Database (*.db *.sqlite *.sqlite3);;All Files (*)",
        )
        if selected:
            self._txt_vector_db_path.setText(selected)

    def _build_current_app_config(self) -> AppConfig:
        """Build AppConfig from current text field inputs.

        Returns:
            AppConfig instance populated from UI inputs.
        """
        data_dir_str = self._txt_data_dir.text().strip() or "data"
        db_path_str = self._txt_db_path.text().strip() or "database/tax_data.db"
        vector_db_str = self._txt_vector_db_path.text().strip() or "database/tax_vectors.db"

        return AppConfig(
            data_dir=Path(data_dir_str).resolve(),
            db_path=Path(db_path_str).resolve(),
            vector_db_path=Path(vector_db_str).resolve(),
            sessions_base_dir=self._config.sessions_base_dir,
        )

    def _save_and_apply(self) -> None:
        """Save settings to ~/.config/tax-return-ai/config.json and hot-reload."""
        app_config = self._build_current_app_config()
        saved_file = save_user_config(app_config)
        set_app_config(app_config)

        new_ui_config = self._config.with_app_config(app_config)
        self._config = new_ui_config
        self.config_updated.emit(new_ui_config)

        self._lbl_action_status.setText(
            f"✅ Saved to {saved_file.name} and applied successfully!"
        )
        QMessageBox.information(
            self,
            "Settings Saved",
            f"Configuration successfully saved to:\n{saved_file}\n\nActive database and data directory reloaded.",
        )

    def _apply_hot_reload(self) -> None:
        """Hot-reload current session with active settings without saving to disk."""
        app_config = self._build_current_app_config()
        set_app_config(app_config)

        new_ui_config = self._config.with_app_config(app_config)
        self._config = new_ui_config
        self.config_updated.emit(new_ui_config)

        self._lbl_action_status.setText("✅ Settings applied to current session!")

    def _reset_defaults(self) -> None:
        """Reset inputs to defaults (ignoring user config file)."""
        default_config = load_config(use_user_config=False)
        self._load_from_config(default_config)
        self._lbl_action_status.setText("↺ Reset to default paths. Click 'Save' or 'Apply' to activate.")

    def reload_config(self, config: UIConfig) -> None:
        """Reload configuration state in SettingsTab.

        Args:
            config: Newly applied UIConfig.
        """
        self._config = config
        self._load_from_config(config.app_config)
