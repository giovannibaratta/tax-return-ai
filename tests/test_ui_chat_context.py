"""Tests for Chat UI context document reference feature (@ trigger, +context button, character threshold warning, and verbatim prompt assembly)."""

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from backend.chat.context_loader import format_prompt_with_context, load_context_doc
from backend.chat.models import AssistantChatMessage
from backend.db_manager import DatabaseManager, LocalDb
from backend.utils.agents import ToolCallInfo
from src.ui.chat_tab import ChatTab
from src.ui.context_file_selector import ContextFileSelectorWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure single QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data folder hierarchy with sample documents."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    scenarios_dir = data_dir / "scenarios"
    scenarios_dir.mkdir()

    doc1 = scenarios_dir / "tax_scenario_2024.json"
    doc1.write_text('{"year": 2024, "tax_due": 1500}', encoding="utf-8")

    doc2 = data_dir / "large_record.txt"
    doc2.write_text("A" * 2500, encoding="utf-8")
    return data_dir

@pytest.fixture
def db_instance(tmp_path):
    """Provide isolated DatabaseManager fixture."""
    db_path = str(tmp_path / "test_chat_context.db")
    db = DatabaseManager(db_config=LocalDb(db_path=db_path, vector_db_path=str(Path(db_path).parent / "vector.db")))
    yield db
    db.close()


def test_context_loader_load_and_format(temp_data_dir: Path):
    # Given: A document in temporary data folder
    doc_path = temp_data_dir / "scenarios" / "tax_scenario_2024.json"

    # When: Document is loaded and formatted into prompt
    doc = load_context_doc(doc_path, base_dir=temp_data_dir)
    prompt = format_prompt_with_context("Calculate tax for scenario", [doc])

    # Then: Verbatim content and attached context header are present
    assert doc.char_count == len('{"year": 2024, "tax_due": 1500}')
    assert "--- Attached Context Documents ---" in prompt
    assert "tax_scenario_2024.json" in prompt
    assert '{"year": 2024, "tax_due": 1500}' in prompt


def test_context_file_selector_tree_and_filter(temp_data_dir: Path, qapp):
    # Given: ContextFileSelectorWidget initialized with temporary data folder
    selector = ContextFileSelectorWidget(data_dir=temp_data_dir)

    # When: User types search query filtering by filename
    selector._search_input.setText("tax_scenario")

    # Then: Tree contains top level item and matching item is visible
    assert selector._tree.topLevelItemCount() > 0
    item = selector._tree.topLevelItem(0)
    assert item is not None
    assert not item.isHidden()
    selector.close()


def test_chat_tab_inline_at_trigger_and_deletion(temp_data_dir: Path, db_instance: DatabaseManager, qapp):
    # Given: ChatTab instance and dummy data document
    chat_tab = ChatTab(db=db_instance)
    doc_path = str(temp_data_dir / "scenarios" / "tax_scenario_2024.json")

    # When: File is selected via inline trigger
    chat_tab._on_inline_file_selected("scenarios/tax_scenario_2024.json", doc_path)

    # Then: Placeholder is inserted into input text edit and stored in inline docs
    assert "@[scenarios/tax_scenario_2024.json]" in chat_tab._input_text.toPlainText()
    assert "scenarios/tax_scenario_2024.json" in chat_tab._inline_attached_docs

    # When: Placeholder is deleted from text edit
    chat_tab._input_text.setText("Ask question without placeholder")

    # Then: Document reference is removed automatically from inline attached docs
    assert "scenarios/tax_scenario_2024.json" not in chat_tab._inline_attached_docs


def test_chat_tab_bottom_context_and_char_warning(temp_data_dir: Path, db_instance: DatabaseManager, qapp):
    # Given: ChatTab and a large document (>2,000 chars)
    chat_tab = ChatTab(db=db_instance)
    chat_tab.show()
    large_doc_path = str(temp_data_dir / "large_record.txt")

    # When: Large document attached via + context button
    chat_tab._on_bottom_file_selected("large_record.txt", large_doc_path)

    # Then: Visual chip is added, and warning label is visible with char count
    assert "large_record.txt" in chat_tab._bottom_attached_docs
    assert not chat_tab._warning_label.isHidden()
    assert "2,500 chars" in chat_tab._warning_label.text()
    assert "exceeds 2,000 char limit" in chat_tab._warning_label.text()

    # When: Bottom document context is removed
    chat_tab._remove_bottom_doc("large_record.txt")

    # Then: Chip removed and warning label hidden
    assert "large_record.txt" not in chat_tab._bottom_attached_docs
    assert chat_tab._warning_label.isHidden()


def test_atomic_tag_deletion_and_chip_close(temp_data_dir: Path, db_instance: DatabaseManager, qapp):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QPushButton

    # Given: ChatTab with a document attached via inline @ and another via + context
    chat_tab = ChatTab(db=db_instance)
    chat_tab.show()
    small_doc_path = str(temp_data_dir / "scenarios" / "tax_scenario_2024.json")

    # When: Document attached via inline @ menu
    chat_tab._on_inline_file_selected("scenarios/tax_scenario_2024.json", small_doc_path)

    # Then: Total chars label is visible even when <= 2k
    assert not chat_tab._warning_label.isHidden()
    assert "31 chars" in chat_tab._warning_label.text()

    # When: Backspace key pressed with cursor at end of @[...] placeholder tag
    cursor = chat_tab._input_text.textCursor()
    cursor.setPosition(len(chat_tab._input_text.toPlainText()))
    chat_tab._input_text.setTextCursor(cursor)

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    chat_tab._input_text.keyPressEvent(event)

    # Then: Entire @[...] placeholder tag is deleted at once
    assert "@[scenarios/tax_scenario_2024.json]" not in chat_tab._input_text.toPlainText()
    assert "scenarios/tax_scenario_2024.json" not in chat_tab._inline_attached_docs

    # When: Document attached via + context button and chip close button clicked
    chat_tab._on_bottom_file_selected("scenarios/tax_scenario_2024.json", small_doc_path)
    assert "scenarios/tax_scenario_2024.json" in chat_tab._bottom_attached_docs

    item = chat_tab._chips_layout.itemAt(0)
    assert item is not None and item.widget() is not None
    chip_widget = item.widget()
    assert chip_widget is not None
    close_btn = chip_widget.findChild(QPushButton)
    assert close_btn is not None
    close_btn.click()

    # Then: Bottom document removed via chip close button click
    assert "scenarios/tax_scenario_2024.json" not in chat_tab._bottom_attached_docs


def test_chat_tab_toggle_tool_calls_visibility(db_instance: DatabaseManager, qapp):
    # Given: ChatTab with active session containing assistant message with tool calls
    chat_tab = ChatTab(db=db_instance)
    chat_tab._on_new_chat_clicked()
    assert chat_tab._active_session is not None

    tool_call = ToolCallInfo(tool_name="calculate_cgt", args={"amount": 1000})
    msg = AssistantChatMessage(
        content="CGT calculated successfully.",
        tool_calls=[tool_call],
    )
    chat_tab._active_session.messages.append(msg)
    chat_tab._render_messages()

    # Then: Tool call toggle button is visible and initially checked ("Tool Calls: Visible")
    assert chat_tab._toggle_tools_btn.isChecked()
    assert chat_tab._toggle_tools_btn.text() == "Tool Calls: Visible"
    labels_with_tool = [lbl for lbl in chat_tab.findChildren(QLabel) if "Tool Used" in lbl.text()]
    assert len(labels_with_tool) == 1

    # When: Toggle button is clicked to hide tool calls
    chat_tab._toggle_tools_btn.click()
    qapp.processEvents()

    # Then: Button state and text update to "Tool Calls: Hidden" and tool call card is removed from render
    assert not chat_tab._toggle_tools_btn.isChecked()
    assert chat_tab._toggle_tools_btn.text() == "Tool Calls: Hidden"
    labels_with_tool_hidden = [lbl for lbl in chat_tab.findChildren(QLabel) if "Tool Used" in lbl.text()]
    assert len(labels_with_tool_hidden) == 0

    # When: Toggle button is clicked again to re-enable visibility
    chat_tab._toggle_tools_btn.click()
    qapp.processEvents()

    # Then: Button state and text return to "Tool Calls: Visible" and tool call card is re-rendered
    assert chat_tab._toggle_tools_btn.isChecked()
    assert chat_tab._toggle_tools_btn.text() == "Tool Calls: Visible"
    labels_with_tool_visible = [lbl for lbl in chat_tab.findChildren(QLabel) if "Tool Used" in lbl.text()]
    assert len(labels_with_tool_visible) == 1
