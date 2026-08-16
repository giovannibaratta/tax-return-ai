"""PySide6 Context File Selector widget for referencing documents in data folder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QIcon, QKeyEvent, QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.chat.context_loader import get_data_dir


class ContextFileSelectorWidget(QFrame):
    """Popup widget with search filter and QTreeWidget showing data folder structure.

    Signals:
        file_selected: Emitted when a document file is selected (relative_path, full_path).
        cancelled: Emitted when popup cancelled / ESC pressed.
    """

    file_selected: Signal = Signal(str, str)
    cancelled: Signal = Signal()

    def __init__(
        self,
        data_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize selector widget.

        Args:
            data_dir: Path to root data directory.
            parent: Parent widget.
        """
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._data_dir = (data_dir or get_data_dir()).resolve()

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
            QLineEdit {
                border: 1px solid #94a3b8;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #f8fafc;
            }
            QTreeWidget {
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QTreeWidget::item:selected {
                background-color: #e0f2fe;
                color: #0369a1;
            }
            """
        )
        self.setFixedSize(360, 280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr_label = QLabel("📂 <b>Select document from data/</b>")
        hdr_label.setStyleSheet("color: #334155;")
        layout.addWidget(hdr_label)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search file by name...")
        self._search_input.textChanged.connect(self._on_search_text_changed)
        self._search_input.installEventFilter(self)
        layout.addWidget(self._search_input)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._tree.itemActivated.connect(self._on_tree_item_activated)
        self._tree.installEventFilter(self)
        layout.addWidget(self._tree)

        self._populate_tree()

    def _populate_tree(self) -> None:
        """Populate tree widget recursively with data directory contents."""
        self._tree.clear()
        if not self._data_dir.exists():
            return

        root_item = QTreeWidgetItem(self._tree, [self._data_dir.name])
        root_item.setData(0, Qt.ItemDataRole.UserRole + 1, "dir")
        root_item.setData(0, Qt.ItemDataRole.UserRole + 2, str(self._data_dir))
        root_item.setIcon(0, QIcon())

        self._build_tree_nodes(self._data_dir, root_item)
        root_item.setExpanded(True)
        self._tree.setCurrentItem(root_item)

    def _build_tree_nodes(self, current_dir: Path, parent_item: QTreeWidgetItem) -> None:
        """Recursively build tree items for current directory.

        Args:
            current_dir: Current directory Path.
            parent_item: Parent QTreeWidgetItem.
        """
        try:
            entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue

            item = QTreeWidgetItem(parent_item, [entry.name])
            if entry.is_dir():
                item.setData(0, Qt.ItemDataRole.UserRole + 1, "dir")
                item.setData(0, Qt.ItemDataRole.UserRole + 2, str(entry))
                item.setText(0, f"📁 {entry.name}")
                self._build_tree_nodes(entry, item)
            else:
                item.setData(0, Qt.ItemDataRole.UserRole + 1, "file")
                item.setData(0, Qt.ItemDataRole.UserRole + 2, str(entry))
                try:
                    rel_path = str(entry.relative_to(self._data_dir.parent))
                except ValueError:
                    rel_path = entry.name
                item.setData(0, Qt.ItemDataRole.UserRole + 3, rel_path)
                item.setText(0, f"📄 {entry.name}")

    def _on_search_text_changed(self, text: str) -> None:
        """Filter tree items based on search query substring match.

        Args:
            text: Query string.
        """
        query = text.strip().lower()

        def filter_item(item: QTreeWidgetItem) -> bool:
            node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
            name = item.text(0).lower()

            child_matches = False
            for i in range(item.childCount()):
                if filter_item(item.child(i)):
                    child_matches = True

            match = (query in name) or child_matches
            item.setHidden(not match)
            if query and match and node_type == "dir":
                item.setExpanded(True)
            return match

        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item is not None:
                filter_item(item)

    def _on_tree_item_activated(self, item: QTreeWidgetItem, column: int = 0) -> None:
        """Handle selection activation (Enter or double-click) of item.

        Args:
            item: Activated QTreeWidgetItem.
            column: Active column.
        """
        node_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if node_type == "file":
            full_path = item.data(0, Qt.ItemDataRole.UserRole + 2)
            rel_path = item.data(0, Qt.ItemDataRole.UserRole + 3)
            self.file_selected.emit(rel_path, full_path)
            self.close()
        elif node_type == "dir":
            item.setExpanded(not item.isExpanded())

    def showEvent(self, event: QShowEvent) -> None:
        """Ensure input widget gains window focus immediately when displayed."""
        super().showEvent(event)
        self.activateWindow()
        self._search_input.setFocus()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Handle key events for keyboard navigation between search input and tree widget.

        Args:
            watched: Object receiving event.
            event: QEvent instance.

        Returns:
            True if event handled, False otherwise.
        """
        if event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            key = event.key()

            if key == Qt.Key.Key_Escape:
                self.cancelled.emit()
                self.close()
                return True

            if watched == self._search_input:
                if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                    self._tree.setFocus()
                    if self._tree.currentItem() is None and self._tree.topLevelItemCount() > 0:
                        top = self._tree.topLevelItem(0)
                        if top is not None:
                            self._tree.setCurrentItem(top)
                    return True
                elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    curr = self._tree.currentItem()
                    if curr is not None:
                        self._on_tree_item_activated(curr)
                    return True

            elif watched == self._tree:
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    curr = self._tree.currentItem()
                    if curr is not None:
                        self._on_tree_item_activated(curr)
                    return True

        return super().eventFilter(watched, event)
