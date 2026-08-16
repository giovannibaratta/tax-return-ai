"""PySide6 Chat Tab for tax return assistant conversation session UI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from backend.chat.agent import ChatDeps
from backend.chat.context_loader import format_prompt_with_context, load_context_doc
from backend.chat.models import (
    AttachedContextDoc,
    ChatMessage,
    ChatSession,
    UserChatMessage,
)
from backend.chat.session_store import SessionStore
from backend.chat.tax_filing_agent import TaxFilingDeps
from backend.db_manager import DatabaseManager
from backend.utils.agents import SharedAgentDeps, ToolCallInfo
from src.jurisdiction.ireland.tax_form_models import IrishTaxFilingSession
from src.ui.chat_input_widgets import ContextChipWidget, ContextTextEdit
from src.ui.context_file_selector import ContextFileSelectorWidget
from src.ui.workers import ChatWorker

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class ChatTab(QWidget):
    """Conversational query tab with session menu, message log, and tool traces."""

    def __init__(
        self,
        db: DatabaseManager,
        session_store: SessionStore[Any] | None = None,
        agent: Agent[ChatDeps, str] | Agent[TaxFilingDeps, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize ChatTab.

        Args:
            db: DatabaseManager instance.
            session_store: Optional SessionStore instance.
            agent: Optional pre-configured PydanticAI agent instance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._db = db
        self._store: SessionStore[Any] = session_store if session_store is not None else SessionStore[ChatSession]()
        self._agent: Agent[ChatDeps, str] | Agent[TaxFilingDeps, str] | None = agent
        self._active_session: ChatSession | None = None
        self._worker: ChatWorker | None = None

        self._inline_attached_docs: dict[str, AttachedContextDoc] = {}
        self._bottom_attached_docs: dict[str, AttachedContextDoc] = {}
        self._selector_popup: ContextFileSelectorWidget | None = None
        self._show_tool_calls: bool = True

        self._init_ui()
        self._load_sessions_list()

    def hide_sidebar(self) -> None:
        """Hide the left sidebar session list."""
        self._sidebar.hide()

    def _init_ui(self) -> None:
        """Build layout with splitter separating left session menu and right chat pane."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -------------------------------------------------------------------
        # Left Panel: Session Sidebar Menu
        # -------------------------------------------------------------------
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        new_btn = QPushButton("+ New Chat")
        new_btn.setFixedHeight(34)
        font = new_btn.font()
        font.setBold(True)
        new_btn.setFont(font)
        new_btn.clicked.connect(self._on_new_chat_clicked)
        sidebar_layout.addWidget(new_btn)

        sidebar_lbl = QLabel("Recent Sessions")
        sidebar_lbl.setStyleSheet("color: gray; font-weight: bold; margin-top: 6px;")
        sidebar_layout.addWidget(sidebar_lbl)

        self._session_list = QListWidget()
        self._session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._session_list.customContextMenuRequested.connect(self._show_session_context_menu)
        self._session_list.itemClicked.connect(self._on_session_item_clicked)
        sidebar_layout.addWidget(self._session_list)

        self._sidebar = sidebar
        self._sidebar.setMinimumWidth(220)
        self._sidebar.setMaximumWidth(320)
        splitter.addWidget(self._sidebar)

        # -------------------------------------------------------------------
        # Right Panel: Active Chat Session Interface
        # -------------------------------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 4, 8, 4)

        # Session Header & Toolbar
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self._header_label = QLabel("Select or start a chat session")
        header_font = self._header_label.font()
        header_font.setPointSize(14)
        header_font.setBold(True)
        self._header_label.setFont(header_font)
        header_layout.addWidget(self._header_label, stretch=1)

        self._toggle_tools_btn = QPushButton("Tool Calls: Visible")
        self._toggle_tools_btn.setCheckable(True)
        self._toggle_tools_btn.setChecked(True)
        self._toggle_tools_btn.setToolTip("Toggle displaying tool execution cards in chat messages")
        self._toggle_tools_btn.setStyleSheet(
            "QPushButton { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 10px; color: #334155; font-weight: bold; }"
            "QPushButton:checked { background-color: #e2e8f0; color: #0f172a; border-color: #94a3b8; }"
            "QPushButton:hover { background-color: #cbd5e1; }"
        )
        self._toggle_tools_btn.clicked.connect(self._on_toggle_tools_clicked)
        header_layout.addWidget(self._toggle_tools_btn)

        right_layout.addWidget(header_widget)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        right_layout.addWidget(line)

        # Message Scroll Area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_container = QWidget()
        self._messages_layout = QVBoxLayout(self._scroll_container)
        self._messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._messages_layout.setSpacing(12)
        self._scroll_area.setWidget(self._scroll_container)
        right_layout.addWidget(self._scroll_area, stretch=1)

        # Status Label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #0066cc; font-style: italic;")
        self._status_label.setVisible(False)
        right_layout.addWidget(self._status_label)

        # Context character limit warning label
        self._warning_label = QLabel("")
        self._warning_label.setStyleSheet(
            "background-color: #fefce8; color: #a16207; border: 1px solid #fde047; "
            "border-radius: 4px; padding: 4px 8px; font-weight: bold;"
        )
        self._warning_label.setVisible(False)
        right_layout.addWidget(self._warning_label)

        # Bottom context chips bar & + context button row
        context_bar = QWidget()
        context_bar_layout = QHBoxLayout(context_bar)
        context_bar_layout.setContentsMargins(0, 2, 0, 2)
        context_bar_layout.setSpacing(6)

        self._add_context_btn = QPushButton("+ context")
        self._add_context_btn.setFixedHeight(28)
        self._add_context_btn.setStyleSheet(
            "QPushButton { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 2px 8px; color: #334155; font-weight: bold; }"
            "QPushButton:hover { background-color: #e2e8f0; color: #0f172a; }"
        )
        self._add_context_btn.clicked.connect(self._on_add_context_clicked)
        context_bar_layout.addWidget(self._add_context_btn)

        self._chips_container = QWidget()
        self._chips_layout = QHBoxLayout(self._chips_container)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(4)
        context_bar_layout.addWidget(self._chips_container, stretch=1)

        right_layout.addWidget(context_bar)

        # Input Row
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 4, 0, 0)

        self._input_text = ContextTextEdit()
        self._input_text.setPlaceholderText(
            "Ask a question about tax regulations or calculations... (Type @ to attach document)"
        )
        self._input_text.setMaximumHeight(75)
        self._input_text.at_triggered.connect(self._on_at_triggered)
        self._input_text.textChanged.connect(self._on_input_text_changed)
        input_layout.addWidget(self._input_text, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setMinimumHeight(45)
        self._send_btn.setMinimumWidth(70)
        self._send_btn.clicked.connect(self._on_send_clicked)
        input_layout.addWidget(self._send_btn)

        right_layout.addWidget(input_container)

        splitter.addWidget(right_panel)
        splitter.setSizes([240, 760])

        main_layout.addWidget(splitter)

    # -----------------------------------------------------------------------
    # Session list management
    # -----------------------------------------------------------------------
    def _load_sessions_list(self) -> None:
        """Reload session list from disk and update sidebar."""
        self._session_list.clear()
        sessions = self._store.list_sessions()
        for session in sessions:
            item = QListWidgetItem(session.title)
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self._session_list.addItem(item)

        if sessions and self._active_session is None:
            self._select_session(sessions[0].id)
        elif self._active_session is None:
            new_session = self._store.create_session(title="New Chat", auto_save=False)
            self._active_session = new_session
            self._header_label.setText(new_session.title)
            self._render_messages()

    def _select_session(self, session_id: str) -> None:
        """Select active session and update chat log display.

        Args:
            session_id: Target session ID.
        """
        session = self._store.load_session(session_id)
        if session is None:
            return

        self._active_session = session
        self._header_label.setText(session.title)
        self._render_messages()

    def _on_new_chat_clicked(self) -> None:
        """Create new in-memory session without immediately writing to disk."""
        new_session = self._store.create_session(title="New Chat", auto_save=False)
        self._active_session = new_session
        self._header_label.setText(new_session.title)
        self._session_list.clearSelection()
        self._render_messages()

    def _on_session_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle selection of session from sidebar list."""
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id:
            self._select_session(session_id)

    def _show_session_context_menu(self, pos: QPoint) -> None:
        """Context menu for renaming or deleting sessions."""
        item = self._session_list.itemAt(pos)
        if item is None:
            return

        session_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)

        rename_act = menu.addAction("Rename Chat")
        delete_act = menu.addAction("Delete Chat")

        action = menu.exec_(self._session_list.mapToGlobal(pos))
        if action == rename_act:
            self._rename_session(session_id)
        elif action == delete_act:
            self._delete_session(session_id)

    def _rename_session(self, session_id: str) -> None:
        """Prompt user for new title and rename session."""
        session = self._store.load_session(session_id)
        if session is None:
            return
        new_title, ok = QInputDialog.getText(self, "Rename Session", "Enter new chat title:", text=session.title)
        if ok and new_title.strip():
            session.title = new_title.strip()
            self._store.save_session(session)
            self._load_sessions_list()
            if self._active_session and self._active_session.id == session_id:
                self._header_label.setText(session.title)

    def _delete_session(self, session_id: str) -> None:
        """Confirm and delete session."""
        res = QMessageBox.question(
            self,
            "Delete Session",
            "Are you sure you want to delete this chat session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if res == QMessageBox.StandardButton.Yes:
            self._store.delete_session(session_id)
            if self._active_session and self._active_session.id == session_id:
                self._active_session = None
            self._load_sessions_list()

    # -----------------------------------------------------------------------
    # Message log rendering & interaction
    # -----------------------------------------------------------------------
    def _clear_messages_layout(self) -> None:
        """Remove all widgets from scroll container layout."""
        while self._messages_layout.count() > 0:
            item = self._messages_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

    def _render_messages(self) -> None:
        """Render all messages for active session."""
        self._clear_messages_layout()

        if self._active_session is None:
            return

        for msg in self._active_session.messages:
            self._add_message_widget(msg)

    def _add_message_widget(self, msg: ChatMessage) -> None:
        """Add single message widget to chat scroll container."""
        card = QFrame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)

        if msg.role == "user":
            card.setStyleSheet("QFrame { background-color: #eef5ff; border: 1px solid #cce0ff; border-radius: 8px; }")
            role_label = QLabel("<b>User</b>")
        elif msg.role == "assistant":
            card.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #e2e8f0; border-radius: 8px; }")
            role_label = QLabel("<b>Assistant</b>")
        else:
            card.setStyleSheet("QFrame { background-color: #fff8e6; border: 1px solid #ffe0b2; border-radius: 8px; }")
            role_label = QLabel(f"<b>{msg.role.capitalize()}</b>")

        card_layout.addWidget(role_label)

        # Tool calls traces if assistant used tools and tool calls visibility is enabled
        if self._show_tool_calls:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_box = self._build_tool_call_card(tool_call)
                    card_layout.addWidget(tool_box)

        content_lbl = QLabel()
        content_lbl.setTextFormat(Qt.TextFormat.MarkdownText)
        content_lbl.setText(msg.content)
        content_lbl.setWordWrap(True)
        content_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card_layout.addWidget(content_lbl)

        # Token usage footer for assistant messages
        if msg.role == "assistant":
            usage = getattr(msg, "usage", None)
            if usage:
                req = getattr(usage, "request_tokens", 0)
                resp = getattr(usage, "response_tokens", 0)
                tot = getattr(usage, "total_tokens", 0)
                cached = getattr(usage, "cached_tokens", 0)
                usage_str = f"⚡ Tokens: {req:,} prompt | {resp:,} completion | {tot:,} total"
                if cached > 0:
                    usage_str += f" | {cached:,} cached"
                usage_lbl = QLabel(usage_str)
                usage_lbl.setStyleSheet("color: #64748b; font-size: 11px; margin-top: 4px;")
                usage_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                card_layout.addWidget(usage_lbl)

        self._messages_layout.addWidget(card)

    def _on_toggle_tools_clicked(self, checked: bool) -> None:
        """Toggle visibility of tool call execution cards in chat messages.

        Args:
            checked: True to display tool call traces, False to hide them.
        """
        self._show_tool_calls = checked
        if checked:
            self._toggle_tools_btn.setText("Tool Calls: Visible")
        else:
            self._toggle_tools_btn.setText("Tool Calls: Hidden")
        self._render_messages()

    def update_font_size(self, font_size: int) -> None:
        """Update font size dynamically across chat tab elements.

        Args:
            font_size: Point size for font.
        """
        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)
        self._session_list.setFont(font)
        self._input_text.setFont(font)
        self._send_btn.setFont(font)
        self._toggle_tools_btn.setFont(font)
        self._status_label.setFont(font)
        self._render_messages()

    def _build_tool_call_card(self, tool_call: ToolCallInfo) -> QWidget:
        """Build styled widget representing a tool call and accessed resources.

        Args:
            tool_call: ToolCallInfo dataclass.

        Returns:
            Formatted QFrame widget.
        """
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px dashed #94a3b8; border-radius: 6px; padding: 6px; margin: 4px 0; }"
        )
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 4, 6, 4)

        header = QLabel(f"🔧 <b>Tool Used:</b> <code>{tool_call.tool_name}</code>")
        header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box_layout.addWidget(header)

        if tool_call.args:
            args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.args.items())
            args_lbl = QLabel(f"<b>Args:</b> {args_str}")
            args_lbl.setStyleSheet("color: #475569;")
            args_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box_layout.addWidget(args_lbl)

        if tool_call.result_summary:
            sum_lbl = QLabel(f"<b>Result:</b> {tool_call.result_summary}")
            sum_lbl.setStyleSheet("color: #334155;")
            sum_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box_layout.addWidget(sum_lbl)

        if tool_call.resources:
            res_lbl = QLabel("<b>Accessed Resources:</b>")
            res_lbl.setStyleSheet("color: #1e293b; font-weight: bold;")
            res_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box_layout.addWidget(res_lbl)

            for res in tool_call.resources:
                if getattr(res, "resource_type", "") == "financial_record" or hasattr(res, "record_id"):
                    rec_id = getattr(res, "record_id", None)
                    action = getattr(res, "action", None)
                    sym_isin = getattr(res, "symbol", None) or getattr(res, "isin", None)
                    tot = getattr(res, "quantity", None) or getattr(res, "total_amount", None)
                    juris = getattr(res, "jurisdiction", None)
                    file_name = getattr(res, "source_file_name", None)

                    res_text = f"• Financial Record #{rec_id}"
                    if action:
                        res_text += f": {action.upper()}"
                    if sym_isin:
                        res_text += f" ({sym_isin})"
                    if tot is not None:
                        res_text += f" — Amount: {tot}"
                    if juris:
                        res_text += f" [{juris.upper()}]"
                    if file_name:
                        res_text += f" ({file_name})"
                elif getattr(res, "resource_type", "") == "document_page":
                    doc_name = getattr(res, "document_name", "Document")
                    page = getattr(res, "page_number", "?")
                    total_p = getattr(res, "total_pages", "?")
                    juris = getattr(res, "jurisdiction", None)
                    res_text = f"• 📄 Full Page {page}/{total_p} of {doc_name}"
                    if juris:
                        res_text += f" [{juris.upper()}]"
                else:
                    doc_name = getattr(res, "document_name", "Document")
                    res_text = f"• {doc_name}"
                    juris = getattr(res, "jurisdiction", None)
                    page = getattr(res, "page_number", None)
                    chunk_id = getattr(res, "chunk_id", None)
                    if juris:
                        res_text += f" [{juris.upper()}]"
                    if page:
                        res_text += f" (p. {page})"
                    if chunk_id:
                        res_text += f" [Chunk #{chunk_id}]"

                r_item = QLabel(res_text)
                r_item.setStyleSheet("color: #0f172a; margin-left: 8px;")
                r_item.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                box_layout.addWidget(r_item)

                snippet = getattr(res, "snippet", None)
                if snippet:
                    snip_lbl = QLabel(f'  <i>"{snippet}..."</i>')
                    snip_lbl.setStyleSheet("color: #64748b; margin-left: 14px;")
                    snip_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    box_layout.addWidget(snip_lbl)

        return box

    # -----------------------------------------------------------------------
    # Document context attachments (@ trigger & + context button)
    # -----------------------------------------------------------------------
    def _on_at_triggered(self, pos: QPoint) -> None:
        """Open context file selector near typing position when '@' key typed.

        Args:
            pos: Global position near cursor.
        """
        self._show_file_selector(pos, is_inline=True)

    def _on_add_context_clicked(self) -> None:
        """Open context file selector near '+ context' button."""
        btn_pos = self._add_context_btn.mapToGlobal(QPoint(0, self._add_context_btn.height()))
        self._show_file_selector(btn_pos, is_inline=False)

    def _show_file_selector(self, global_pos: QPoint, is_inline: bool) -> None:
        """Display ContextFileSelectorWidget popup at target position.

        Args:
            global_pos: Global screen coordinates for popup.
            is_inline: True if triggered via '@' inline, False if via '+ context'.
        """
        if self._selector_popup is not None:
            self._selector_popup.close()

        self._selector_popup = ContextFileSelectorWidget(parent=self)
        if is_inline:
            self._selector_popup.file_selected.connect(self._on_inline_file_selected)
        else:
            self._selector_popup.file_selected.connect(self._on_bottom_file_selected)

        self._selector_popup.move(global_pos)
        self._selector_popup.show()
        self._selector_popup.raise_()

    def _on_inline_file_selected(self, rel_path: str, full_path: str) -> None:
        """Handle inline document selection from '@' menu.

        Args:
            rel_path: Relative path string.
            full_path: Absolute file path.
        """
        doc = load_context_doc(full_path)
        key = rel_path or doc.relative_path
        doc.relative_path = key
        self._inline_attached_docs[key] = doc

        cursor = self._input_text.textCursor()
        # Remove trailing '@' typed before inserting tag
        text_before = self._input_text.toPlainText()
        if text_before and text_before.endswith("@"):
            cursor.deletePreviousChar()

        cursor.insertText(f"@[{key}] ")
        self._input_text.setTextCursor(cursor)
        self._update_context_warning()

    def _on_bottom_file_selected(self, rel_path: str, full_path: str) -> None:
        """Handle bottom document selection from '+ context' button.

        Args:
            rel_path: Relative path string.
            full_path: Absolute file path.
        """
        doc = load_context_doc(full_path)
        key = rel_path or doc.relative_path
        doc.relative_path = key
        self._bottom_attached_docs[key] = doc
        self._render_bottom_chips()
        self._update_context_warning()

    def _remove_bottom_doc(self, rel_path: str) -> None:
        """Remove a document attached via '+ context' button.

        Args:
            rel_path: Target relative path.
        """
        self._bottom_attached_docs.pop(rel_path, None)
        self._render_bottom_chips()
        self._update_context_warning()

    def _render_bottom_chips(self) -> None:
        """Re-render visual pill badges in bottom context container."""
        while self._chips_layout.count() > 0:
            item = self._chips_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        for doc in self._bottom_attached_docs.values():
            chip = ContextChipWidget(doc, parent=self._chips_container)
            chip.remove_requested.connect(self._remove_bottom_doc)
            self._chips_layout.addWidget(chip)

    def _on_input_text_changed(self) -> None:
        """Sync inline attached docs with present '@[path]' placeholders in input text."""
        current_text = self._input_text.toPlainText()
        removed_paths = [rel_path for rel_path in self._inline_attached_docs if f"@[{rel_path}]" not in current_text]

        for rel_path in removed_paths:
            self._inline_attached_docs.pop(rel_path, None)

        self._update_context_warning()

    def _update_context_warning(self) -> None:
        """Update context total character count indicator and threshold warning label."""
        total_chars = sum(d.char_count for d in self._inline_attached_docs.values()) + sum(
            d.char_count for d in self._bottom_attached_docs.values()
        )
        doc_count = len(self._inline_attached_docs) + len(self._bottom_attached_docs)
        threshold = 2000

        if doc_count == 0:
            self._warning_label.setVisible(False)
        elif total_chars > threshold:
            self._warning_label.setText(
                f"⚠️ Context total: {total_chars:,} chars ({doc_count} docs) — exceeds {threshold:,} char limit!"
            )
            self._warning_label.setStyleSheet(
                "background-color: #fefce8; color: #a16207; border: 1px solid #fde047; "
                "border-radius: 4px; padding: 4px 8px; font-weight: bold;"
            )
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setText(f"📊 Context total: {total_chars:,} chars ({doc_count} docs)")
            self._warning_label.setStyleSheet(
                "background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; "
                "border-radius: 4px; padding: 4px 8px; font-weight: 500;"
            )
            self._warning_label.setVisible(True)

    # -----------------------------------------------------------------------
    # Sending turns
    # -----------------------------------------------------------------------
    def send_user_message(self, text: str) -> None:
        """Programmatically submit a user message turn."""
        self._input_text.setPlainText(text)
        self._on_send_clicked()

    def _on_send_clicked(self) -> None:
        """Submit new message turn with verbatim document context."""
        raw_text = self._input_text.toPlainText().strip()
        if not raw_text and not self._inline_attached_docs and not self._bottom_attached_docs:
            return

        if self._active_session is None:
            self._on_new_chat_clicked()

        assert self._active_session is not None

        all_docs = list(self._inline_attached_docs.values()) + list(self._bottom_attached_docs.values())
        full_prompt = format_prompt_with_context(raw_text, all_docs)

        # Add user message
        user_msg = UserChatMessage(content=raw_text)
        self._active_session.messages.append(user_msg)

        # Update title if first message
        if len(self._active_session.messages) == 1:
            self._active_session.title = raw_text[:30] + ("..." if len(raw_text) > 30 else "")
            self._header_label.setText(self._active_session.title)

        self._store.save_session(self._active_session)
        self._render_messages()

        self._input_text.clear()
        self._inline_attached_docs.clear()
        self._bottom_attached_docs.clear()
        self._render_bottom_chips()
        self._update_context_warning()

        # Disable input while waiting
        self._set_input_enabled(False)
        self._status_label.setText("⏳ Agent starting...")
        self._status_label.setVisible(True)

        # Construct session-specific dependencies
        deps: SharedAgentDeps
        if isinstance(self._active_session, IrishTaxFilingSession):
            deps = TaxFilingDeps(
                db=self._db,
                form_state=self._active_session.form_state,
            )
        else:
            deps = ChatDeps(
                db=self._db,
            )

        # Run async worker with full prompt containing verbatim context
        self._worker = ChatWorker(
            prompt=full_prompt,
            past_messages=self._active_session.messages[:-1],
            deps=deps,
            agent=self._agent,
            parent=self,
        )
        self._worker.message_ready.connect(self._on_worker_message_ready)
        self._worker.progress_updated.connect(self._on_worker_progress)
        self._worker.approval_requested.connect(self._on_worker_approval_requested)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()

    def _set_input_enabled(self, enabled: bool) -> None:
        """Enable or disable text input and send button."""
        self._input_text.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)

    def _on_worker_progress(self, status: str) -> None:
        """Update status bar with intermediate agent tool interaction progress."""
        self._status_label.setText(f"⏳ {status}")
        self._status_label.setVisible(True)

    def _on_worker_approval_requested(self, current_limit: int, sync_obj: tuple[Any, list[bool]]) -> None:
        """Prompt user on main thread to approve extending request limit.

        Args:
            current_limit: Current tool request limit that was exceeded.
            sync_obj: Tuple of (threading.Event, list[bool]) container.
        """
        event, result_container = sync_obj
        res = QMessageBox.question(
            self,
            "Request Limit Exceeded",
            f"The agent has executed {current_limit} tool requests for this query.\n\n"
            f"Would you like to approve continuing for 50 additional tool requests?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        result_container[0] = res == QMessageBox.StandardButton.Yes
        event.set()

    def _on_worker_message_ready(
        self, assistant_msg: ChatMessage, tool_traces: list[ToolCallInfo], usage_info: Any = None
    ) -> None:
        """Handle completion of agent response."""
        self._set_input_enabled(True)

        usage = usage_info or getattr(assistant_msg, "usage", None)
        if usage:
            req = getattr(usage, "request_tokens", 0)
            resp = getattr(usage, "response_tokens", 0)
            tot = getattr(usage, "total_tokens", 0)
            cached = getattr(usage, "cached_tokens", 0)
            status_text = f"⚡ Turn Tokens: {req:,} prompt | {resp:,} completion | {tot:,} total"
            if cached > 0:
                status_text += f" | {cached:,} cached"
            self._status_label.setText(status_text)
            self._status_label.setVisible(True)
        else:
            self._status_label.setVisible(False)

        if self._active_session is not None:
            self._active_session.messages.append(assistant_msg)
            self._store.save_session(self._active_session)
            self._render_messages()
            self._load_sessions_list()

    def _on_worker_error(self, err_msg: str) -> None:
        """Handle error during worker turn by displaying an ephemeral warning bubble in chat."""
        self._status_label.setVisible(False)
        self._set_input_enabled(True)
        logger.error("Chat turn failed: %s", err_msg)

        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; }")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)

        hdr = QLabel("⚠️ <b>Agent not available</b>")
        hdr.setStyleSheet("color: #991b1b;")
        card_layout.addWidget(hdr)

        msg_lbl = QLabel("The model or agent is currently unavailable or not responding. Please check model status.")
        msg_lbl.setStyleSheet("color: #7f1d1d;")
        msg_lbl.setWordWrap(True)
        card_layout.addWidget(msg_lbl)

        self._messages_layout.addWidget(card)
