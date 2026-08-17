"""Custom input widgets for chat tab context attachments."""

from __future__ import annotations

import re

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QWidget,
)

from backend.chat.models import AttachedContextDoc

# Regular expression matching inline context reference tokens in the format @[filename.pdf].
# Used to enable atomic deletion (treating the entire token as a single unit on backspace/delete).
CONTEXT_TAG_PATTERN: re.Pattern[str] = re.compile(r"@\[[^\]]*\]\s?")


class ContextTextEdit(QTextEdit):
    """QTextEdit with inline '@' trigger detection and atomic placeholder deletion.

    Signals:
        at_triggered: Emitted with global QPoint near cursor position when '@' typed.
    """

    at_triggered: Signal = Signal(QPoint)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Intercept keypress to detect '@' typing and handle atomic tag deletion.

        Args:
            event: QKeyEvent instance.
        """
        key = event.key()

        if key == Qt.Key.Key_At or event.text() == "@":
            cursor = self.textCursor()
            pos = cursor.position()
            text = self.toPlainText()
            # Only trigger popup at start of text or when preceded by whitespace (avoids emails)
            is_word_boundary = pos == 0 or (pos <= len(text) and text[pos - 1].isspace())
            if is_word_boundary:
                cursor_rect = self.cursorRect()
                global_pos = self.viewport().mapToGlobal(cursor_rect.bottomLeft())
                self.at_triggered.emit(global_pos)
            super().keyPressEvent(event)
            return

        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            cursor = self.textCursor()
            if not cursor.hasSelection():
                pos = cursor.position()
                text = self.toPlainText()
                for match in CONTEXT_TAG_PATTERN.finditer(text):
                    start, end = match.span()
                    if key == Qt.Key.Key_Backspace and start < pos <= end:
                        cursor.setPosition(start)
                        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                        cursor.removeSelectedText()
                        return
                    elif key == Qt.Key.Key_Delete and start <= pos < end:
                        cursor.setPosition(start)
                        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                        cursor.removeSelectedText()
                        return

        super().keyPressEvent(event)


class ContextChipWidget(QFrame):
    """Visual badge widget representing a bottom attached document context.

    Signals:
        remove_requested: Emitted with relative_path when 'x' delete button clicked.
    """

    remove_requested: Signal = Signal(str)

    def __init__(self, doc: AttachedContextDoc, parent: QWidget | None = None) -> None:
        """Initialize chip widget.

        Args:
            doc: AttachedContextDoc instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.doc = doc
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #eff6ff;
                border: 1px solid #93c5fd;
                border-radius: 12px;
                padding: 2px 6px;
            }
            QLabel {
                color: #1e40af;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton {
                border: none;
                background: transparent;
                color: #64748b;
                font-weight: bold;
                font-size: 12px;
                padding: 0 2px;
            }
            QPushButton:hover {
                color: #ef4444;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        icon_lbl = QLabel(f"📄 <b>{doc.relative_path}</b> <i>({doc.char_count:,} chars)</i>")
        layout.addWidget(icon_lbl)

        close_btn = QPushButton("✕")
        close_btn.setToolTip("Remove document context")
        target_path = doc.relative_path
        close_btn.clicked.connect(lambda _checked=False, path=target_path: self.remove_requested.emit(path))
        layout.addWidget(close_btn)
