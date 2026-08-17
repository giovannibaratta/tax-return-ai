"""Main window for the Tax Return AI desktop application.

Tab layout (extensible):
    0 – Regulations   : Ingest PDFs + inspect produced chunks.
    [future] Chat     : Conversational query interface.
    [future] Records  : Browse ingested financial records.
"""

from __future__ import annotations

import os
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.db_manager import DatabaseManager
from backend.db_models import TaxDocumentMetadata
from backend.deliberation.models import EvidenceChunk
from backend.domain_models import IngestionDocumentSummary
from src.ui.base_tab import BaseAppTab
from src.ui.chat_tab import ChatTab
from src.ui.classification_tab import AssetClassificationTab
from src.ui.config import UIConfig
from src.ui.profile_tab import TaxpayerProfileTab
from src.ui.records_tab import FinancialRecordsTab
from src.ui.report_tab import IrishTaxReportTab
from src.ui.settings_tab import SettingsTab
from src.ui.workers import IngestionWorker, SearchWorker

# ---------------------------------------------------------------------------
# Regulations tab
# ---------------------------------------------------------------------------


class RegulationsTab(BaseAppTab):
    """Two-panel tab: left side ingests PDFs, right side inspects chunks."""

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        """Initialise the regulations tab.

        Args:
            db: Shared DatabaseManager instance (created once at app startup).
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._db = db
        self._worker: IngestionWorker | None = None
        self._search_worker: SearchWorker | None = None
        self._embedding_runner = None
        self._selected_doc: IngestionDocumentSummary | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(1)
        root.addWidget(splitter)

        splitter.addWidget(self._build_ingest_panel())
        splitter.addWidget(self._build_inspect_panel())
        splitter.setSizes([420, 580])

    def reload_config(self, config: UIConfig) -> None:
        """Reload configuration state and database connections in RegulationsTab.

        Args:
            config: Newly applied UIConfig instance.
        """
        self._db = config.db
        self._load_documents()

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _build_ingest_panel(self) -> QWidget:
        """Build the left panel: file picker + jurisdiction + log."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 8, 12)
        layout.setSpacing(8)

        layout.addWidget(_section_label("Ingest Regulations"))

        # File list display
        self._file_list = QListWidget()
        self._file_list.setFixedHeight(110)
        self._file_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self._file_list)

        # File picker row
        pick_row = QHBoxLayout()
        btn_pick = QPushButton("Select PDFs…")
        btn_pick.clicked.connect(self._pick_files)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_files)
        pick_row.addWidget(btn_pick)
        pick_row.addWidget(btn_clear)
        layout.addLayout(pick_row)

        # Jurisdiction selector
        jur_row = QHBoxLayout()
        jur_row.addWidget(QLabel("Jurisdiction:"))
        self._combo_jurisdiction = QComboBox()
        self._combo_jurisdiction.addItems(["ireland", "italy"])
        jur_row.addWidget(self._combo_jurisdiction)
        jur_row.addStretch()
        layout.addLayout(jur_row)

        # Ingest button
        self._btn_ingest = QPushButton("Ingest")
        self._btn_ingest.setEnabled(False)
        self._btn_ingest.clicked.connect(self._start_ingestion)
        layout.addWidget(self._btn_ingest)

        # Divider
        layout.addWidget(_divider())

        # Log output
        layout.addWidget(_section_label("Log"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Menlo", 10))
        self._log.setMaximumHeight(130)
        layout.addWidget(self._log)

        return panel

    def _build_inspect_panel(self) -> QWidget:
        """Build the right panel: ingested document list + chunk table."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(_section_label("Ingested Documents"))

        # Refresh and Delete Database buttons
        doc_btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._load_documents)
        doc_btn_row.addWidget(btn_refresh)

        btn_delete_db = QPushButton("Delete Database")
        btn_delete_db.clicked.connect(self._delete_entire_db)
        doc_btn_row.addWidget(btn_delete_db)

        layout.addLayout(doc_btn_row)

        # Document list
        self._doc_list = QListWidget()
        self._doc_list.setFixedHeight(160)
        self._doc_list.currentItemChanged.connect(self._on_doc_selected)
        layout.addWidget(self._doc_list)

        # Search bar
        layout.addWidget(_section_label("Semantic Search"))
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search regulations (e.g. 'exit tax rate', 'Quadro RW')...")
        self._search_input.returnPressed.connect(self._run_semantic_search)
        self._btn_search = QPushButton("Search")
        self._btn_search.clicked.connect(self._run_semantic_search)
        btn_clear_search = QPushButton("Clear")
        btn_clear_search.clicked.connect(self._clear_search)
        search_row.addWidget(self._search_input)
        search_row.addWidget(self._btn_search)
        search_row.addWidget(btn_clear_search)
        layout.addLayout(search_row)

        layout.addWidget(_divider())
        layout.addWidget(_section_label("Chunks"))

        # Chunk table
        self._chunk_table = QTableWidget(0, 4)
        self._chunk_table.setHorizontalHeaderLabels(["#", "Page", "Score / SHA", "Text preview"])
        header = self._chunk_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._chunk_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._chunk_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._chunk_table.setAlternatingRowColors(True)
        self._chunk_table.itemSelectionChanged.connect(self._on_chunk_selected)
        layout.addWidget(self._chunk_table)

        layout.addWidget(_divider())

        # Dual text boxes for Child Chunk Text and Parent Full Chunk Text
        text_splitter = QSplitter(Qt.Orientation.Vertical)

        child_widget = QWidget()
        child_layout = QVBoxLayout(child_widget)
        child_layout.setContentsMargins(0, 0, 0, 0)
        child_layout.setSpacing(4)
        child_layout.addWidget(_section_label("Child Chunk Text (Retrieval Unit)"))
        self._chunk_text = QTextEdit()
        self._chunk_text.setReadOnly(True)
        self._chunk_text.setFont(QFont("Georgia", 11))
        child_layout.addWidget(self._chunk_text)

        parent_widget = QWidget()
        parent_layout = QVBoxLayout(parent_widget)
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.setSpacing(4)
        parent_layout.addWidget(_section_label("Parent Chunk Text (Full Context)"))
        self._parent_chunk_text = QTextEdit()
        self._parent_chunk_text.setReadOnly(True)
        self._parent_chunk_text.setFont(QFont("Georgia", 11))
        parent_layout.addWidget(self._parent_chunk_text)

        text_splitter.addWidget(child_widget)
        text_splitter.addWidget(parent_widget)
        text_splitter.setSizes([140, 200])
        layout.addWidget(text_splitter)

        # Load on first paint
        self._load_documents()

        return panel

    # ------------------------------------------------------------------
    # Ingest helpers
    # ------------------------------------------------------------------

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF files", "", "PDF Files (*.pdf)")
        for p in paths:
            if not any(
                self._file_list.item(i).data(Qt.ItemDataRole.UserRole) == p for i in range(self._file_list.count())
            ):
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.ItemDataRole.UserRole, p)
                self._file_list.addItem(item)
        self._btn_ingest.setEnabled(self._file_list.count() > 0)

    def _clear_files(self) -> None:
        self._file_list.clear()
        self._btn_ingest.setEnabled(False)

    def _start_ingestion(self) -> None:
        paths = [self._file_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._file_list.count())]
        self._log.clear()
        self._log.append("Initialising ingestion pipeline…\n")
        self._btn_ingest.setEnabled(False)

        self._worker = IngestionWorker(
            file_paths=paths,
            jurisdiction=self._combo_jurisdiction.currentText(),
            db=self._db,
        )
        self._worker.log_message.connect(self._append_log)
        self._worker.ingest_finished.connect(self._on_ingestion_done)
        self._worker.start()

    def _append_log(self, line: str) -> None:
        self._log.append(line)

    def _on_ingestion_done(self, success: bool) -> None:
        if success:
            self._log.append("\n✓ Ingestion complete.")
            self._load_documents()
        else:
            self._log.append("\n✗ Ingestion failed — see log above.")
        self._btn_ingest.setEnabled(True)

    # ------------------------------------------------------------------
    # Inspect helpers
    # ------------------------------------------------------------------

    def _load_documents(self) -> None:
        try:
            docs = self._db.get_ingested_documents()
        except Exception as exc:
            QMessageBox.critical(self, "DB Error", str(exc))
            return

        self._doc_list.clear()
        for doc in docs:
            label = f"{doc.document_name}  [{doc.jurisdiction}]  ({doc.chunk_count} chunks)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, doc)
            self._doc_list.addItem(item)

    def _on_doc_selected(self, current: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if current is None:
            return
        doc: IngestionDocumentSummary = current.data(Qt.ItemDataRole.UserRole)
        self._selected_doc = doc
        self._load_chunks(doc.document_sha)

    def _load_chunks(self, document_sha: str) -> None:
        try:
            chunks: list[TaxDocumentMetadata] = self._db.get_chunks_for_document(document_sha)
        except Exception as exc:
            QMessageBox.critical(self, "DB Error", str(exc))
            return

        self._chunk_table.setRowCount(0)
        self._chunk_text.clear()
        self._parent_chunk_text.clear()
        for chunk in chunks:
            row = self._chunk_table.rowCount()
            self._chunk_table.insertRow(row)
            self._chunk_table.setItem(row, 0, QTableWidgetItem(str(chunk.chunk_index)))
            self._chunk_table.setItem(row, 1, QTableWidgetItem(str(chunk.page_number)))
            sha_prefix = chunk.document_sha[:12] if chunk.document_sha else ""
            self._chunk_table.setItem(row, 2, QTableWidgetItem(sha_prefix))
            preview = chunk.text_content[:120].replace("\n", " ")
            item = QTableWidgetItem(preview)
            item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "child_text": chunk.text_content,
                    "parent_text": chunk.parent_text_content or chunk.text_content,
                },
            )
            self._chunk_table.setItem(row, 3, item)

    def _on_chunk_selected(self) -> None:
        rows = self._chunk_table.selectedItems()
        if not rows:
            return
        text_item = self._chunk_table.item(self._chunk_table.currentRow(), 3)
        if text_item:
            data = text_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                dict_obj = cast(dict[str, object], data)
                child_val = dict_obj.get("child_text")
                self._chunk_text.setPlainText(str(child_val) if child_val is not None else "")
                parent_val = dict_obj.get("parent_text")
                if isinstance(parent_val, str) and parent_val:
                    self._parent_chunk_text.setPlainText(parent_val)
                else:
                    self._parent_chunk_text.setPlainText("(No parent context available)")
            elif isinstance(data, str):
                self._chunk_text.setPlainText(data)
                self._parent_chunk_text.setPlainText(data)

    def _run_semantic_search(self) -> None:
        """Run embedding search on query entered in the search bar."""
        query = self._search_input.text().strip()
        if not query:
            return

        self._btn_search.setEnabled(False)
        self._log.append(f"\n▶ Running semantic search for: '{query}'…")

        self._search_worker = SearchWorker(
            query=query,
            db=self._db,
            embedding_runner=self._embedding_runner,
            parent=self,
        )
        self._search_worker.results_found.connect(self._on_search_results)
        self._search_worker.error_occurred.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_results(self, results: list[EvidenceChunk]) -> None:
        """Display search results in the chunk table."""
        self._btn_search.setEnabled(True)
        self._log.append(f"✓ Found {len(results)} matching chunks.")

        self._chunk_table.setRowCount(0)
        self._chunk_text.clear()
        self._parent_chunk_text.clear()

        for chunk in results:
            row = self._chunk_table.rowCount()
            self._chunk_table.insertRow(row)
            self._chunk_table.setItem(row, 0, QTableWidgetItem(str(chunk.chunk_index)))
            self._chunk_table.setItem(row, 1, QTableWidgetItem(str(chunk.page_number)))
            dist_str = f"{chunk.distance:.3f}" if chunk.distance is not None else ""
            self._chunk_table.setItem(row, 2, QTableWidgetItem(dist_str))
            preview = f"[{chunk.document_name}] {chunk.text_content[:120]}".replace("\n", " ")
            item = QTableWidgetItem(preview)
            item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "child_text": chunk.text_content,
                    "parent_text": chunk.parent_text_content or chunk.text_content,
                },
            )
            self._chunk_table.setItem(row, 3, item)

    def _on_search_error(self, err_msg: str) -> None:
        """Handle search worker failure."""
        self._btn_search.setEnabled(True)
        self._log.append(f"✗ Search failed:\n{err_msg}")
        QMessageBox.critical(self, "Search Error", err_msg)

    def _clear_search(self) -> None:
        """Clear search query and restore document chunk view."""
        self._search_input.clear()
        self._chunk_table.setRowCount(0)
        self._chunk_text.clear()
        self._parent_chunk_text.clear()
        if self._selected_doc:
            self._load_chunks(self._selected_doc.document_sha)
        else:
            self._load_documents()

    def _delete_entire_db(self) -> None:
        """Prompt user for confirmation with 'delete all' and delete database chunks and vectors."""
        text_input, ok = QInputDialog.getText(
            self,
            "Confirm Database Deletion",
            "WARNING: This will permanently delete all chunks and vectors from the database.\n"
            "This will NOT delete the cache.\n\n"
            "Type 'delete all' to confirm:",
        )
        if not ok:
            return

        if text_input.strip().lower() != "delete all":
            QMessageBox.warning(
                self,
                "Deletion Cancelled",
                "Confirmation text did not match 'delete all'. Database was not deleted.",
            )
            return

        try:
            count = self._db.delete_all_chunks()
            self._log.append(f"\n✓ Deleted {count} chunks and vector embeddings from database.")
            self._load_documents()
            self._chunk_table.setRowCount(0)
            self._chunk_text.clear()
            QMessageBox.information(
                self,
                "Database Deleted",
                f"Successfully deleted {count} chunks and vectors from the database.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "DB Error", str(exc))


# ---------------------------------------------------------------------------
# Placeholder tabs for future sections
# ---------------------------------------------------------------------------


class _PlaceholderTab(QWidget):
    """Minimal placeholder widget shown for tabs not yet implemented."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        lbl = QLabel(f"[{label}]\n\nComing soon.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Helvetica", 14))
        layout.addWidget(lbl)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Root application window with a tab bar for each major section.

    Current tabs:
        Regulations – Ingest + inspect regulation PDFs.

    Planned tabs (placeholders until implemented):
        Chat        – Conversational query interface.
        Records     – Browse ingested financial / transaction records.
    """

    def __init__(
        self,
        config: UIConfig | DatabaseManager | None = None,
        db: DatabaseManager | None = None,
    ) -> None:
        """Create the main window.

        Args:
            config: Centralized UIConfig instance or shared DatabaseManager.
            db: Optional DatabaseManager fallback.

        Raises:
            ValueError: If neither config nor db is provided.
        """
        super().__init__()
        if isinstance(config, UIConfig):
            self._config = config
        elif isinstance(config, DatabaseManager):
            self._config = UIConfig(db=config)
        elif db is not None:
            self._config = UIConfig(db=db)
        else:
            raise ValueError("Either config or db must be provided.")

        self.setWindowTitle("Tax Return AI")
        self.resize(1100, 740)
        self._font_size: int = 13

        self._create_menu_bar()

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._tab_instances: list[BaseAppTab] = [
            RegulationsTab(db=self._config.db),
            IrishTaxReportTab(config=self._config),
            ChatTab(db=self._config.db),
            FinancialRecordsTab(db=self._config.db),
            TaxpayerProfileTab(db=self._config.db),
            AssetClassificationTab(config=self._config),
        ]
        tab_names = [
            "Regulations",
            "Irish Tax Report",
            "Chat",
            "Financial Records",
            "Taxpayer Profile",
            "Asset Classification",
        ]
        for tab, name in zip(self._tab_instances, tab_names):
            self._tabs.addTab(tab, name)

        settings_tab = SettingsTab(config=self._config)
        settings_tab.config_updated.connect(self._on_config_updated)
        self._tabs.addTab(settings_tab, "Settings")
        self._tab_instances.append(settings_tab)

        self.setCentralWidget(self._tabs)

    def _on_config_updated(self, new_config: UIConfig) -> None:
        """Handle dynamic configuration updates from SettingsTab.

        Args:
            new_config: Newly applied UIConfig with updated paths and DatabaseManager.
        """
        self._config = new_config

        for tab in self._tab_instances:
            tab.reload_config(new_config)

        self.statusBar().showMessage("Configuration & databases reloaded successfully.", 5000)

    def _create_menu_bar(self) -> None:
        """Build macOS-style top menu bar with View -> Font size controls."""
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("View")

        zoom_in_action = QAction("Increase Font Size", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self._increase_font_size)

        zoom_out_action = QAction("Decrease Font Size", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self._decrease_font_size)

        reset_zoom_action = QAction("Reset Font Size", self)
        reset_zoom_action.setShortcut("Ctrl+0")
        reset_zoom_action.triggered.connect(self._reset_font_size)

        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)
        view_menu.addSeparator()
        view_menu.addAction(reset_zoom_action)

    def _increase_font_size(self) -> None:
        """Increase global application font size."""
        self._font_size = min(24, self._font_size + 1)
        self._apply_font_size()

    def _decrease_font_size(self) -> None:
        """Decrease global application font size."""
        self._font_size = max(8, self._font_size - 1)
        self._apply_font_size()

    def _reset_font_size(self) -> None:
        """Reset global application font size to default."""
        self._font_size = 13
        self._apply_font_size()

    def _apply_font_size(self) -> None:
        """Apply font size change across application."""
        app = QApplication.instance()
        if isinstance(app, QApplication):
            font = app.font()
            font.setPointSize(self._font_size)
            app.setFont(font)
            self.setFont(font)
            central = self.centralWidget()
            if isinstance(central, QTabWidget):
                central.setFont(font)
                for i in range(central.count()):
                    tab = central.widget(i)
                    if tab is not None and hasattr(tab, "update_font_size"):
                        fn = getattr(tab, "update_font_size")
                        fn(self._font_size)


# ---------------------------------------------------------------------------
# Small layout helpers
# ---------------------------------------------------------------------------


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    font = lbl.font()
    font.setBold(True)
    lbl.setFont(font)
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
