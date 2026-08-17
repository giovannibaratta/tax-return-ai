"""Unit tests for ContextTextEdit and chat input widgets."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from src.ui.chat_input_widgets import ContextTextEdit


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


def test_context_text_edit_at_trigger_at_start(qapp: QApplication) -> None:
    # Given: Empty ContextTextEdit
    widget = ContextTextEdit()
    mock_listener = MagicMock()
    widget.at_triggered.connect(mock_listener)

    # When: Typing '@' as the first character
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_At, Qt.KeyboardModifier.NoModifier, "@")
    widget.keyPressEvent(event)

    # Then: at_triggered signal is emitted
    mock_listener.assert_called_once()


def test_context_text_edit_at_trigger_after_whitespace(qapp: QApplication) -> None:
    # Given: ContextTextEdit with existing text followed by space
    widget = ContextTextEdit()
    widget.setPlainText("Hello ")
    cursor = widget.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    widget.setTextCursor(cursor)

    mock_listener = MagicMock()
    widget.at_triggered.connect(mock_listener)

    # When: Typing '@' after whitespace
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_At, Qt.KeyboardModifier.NoModifier, "@")
    widget.keyPressEvent(event)

    # Then: at_triggered signal is emitted
    mock_listener.assert_called_once()


def test_context_text_edit_at_not_triggered_mid_word(qapp: QApplication) -> None:
    # Given: ContextTextEdit with existing non-whitespace text (e.g. typing email)
    widget = ContextTextEdit()
    widget.setPlainText("user")
    cursor = widget.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    widget.setTextCursor(cursor)

    mock_listener = MagicMock()
    widget.at_triggered.connect(mock_listener)

    # When: Typing '@' immediately following word characters
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_At, Qt.KeyboardModifier.NoModifier, "@")
    widget.keyPressEvent(event)

    # Then: at_triggered signal is NOT emitted
    mock_listener.assert_not_called()


def test_context_text_edit_atomic_tag_backspace(qapp: QApplication) -> None:
    # Given: ContextTextEdit with a context tag
    widget = ContextTextEdit()
    widget.setPlainText("Check @[document.pdf] please")
    cursor = widget.textCursor()
    # Position cursor right after ']' of the tag (position 22)
    cursor.setPosition(22)
    widget.setTextCursor(cursor)

    # When: Pressing Backspace
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(event)

    # Then: The whole context tag token is atomically deleted
    assert widget.toPlainText() == "Check please"
