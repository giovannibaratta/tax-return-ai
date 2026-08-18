"""Irish Tax Report tab containing dynamic tax form and AI filing assistant."""

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.chat.session_store import SessionStore
from backend.chat.tax_filing_agent import create_tax_filing_agent
from src.jurisdiction.ireland.tax_form_models import (
    FormField,
    IrishCG1State,
    IrishForm11State,
    IrishForm12State,
    IrishTaxFilingSession,
    IrishTaxFormState,
    IrishUndeterminedState,
    TaxFilingMetadata,
)
from src.ui.base_tab import BaseAppTab
from src.ui.chat_tab import ChatTab
from src.ui.config import UIConfig


class IrishTaxReportTab(BaseAppTab):
    """A tab containing a dynamic tax form on the left and an AI assistant on the right."""

    def __init__(
        self,
        config: UIConfig,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize IrishTaxReportTab with centralized configuration.

        Args:
            config: Mandatory UIConfig instance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._config = config
        self._db = self._config.db
        self._store = SessionStore(
            sessions_dir=self._config.tax_filing_sessions_dir,
            session_cls=IrishTaxFilingSession,
        )

        self._agent = create_tax_filing_agent()
        self._current_year: int = datetime.now().year - 1
        self._form_state: IrishTaxFormState | None = None
        self._chat_tab: ChatTab | None = None

        self._init_ui()
        self._populate_year_combo()

    def reload_config(self, config: UIConfig) -> None:
        """Reload configuration state and dependencies in IrishTaxReportTab.

        Args:
            config: Newly applied UIConfig instance.
        """
        self._config = config
        self._db = config.db
        self._store = SessionStore(
            sessions_dir=config.tax_filing_sessions_dir,
            session_cls=IrishTaxFilingSession,
        )
        if self._chat_tab is not None:
            self._chat_tab.reload_config(config)

    def _init_ui(self) -> None:
        """Set up the main UI layout."""
        root = QVBoxLayout(self)
        self._init_top_bar(root)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._splitter)

        self._init_form_pane()
        self._init_chat_pane()

        self._splitter.setSizes([400, 400])

        # Timer to poll for form updates
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._refresh_form_view)

    def _init_top_bar(self, root: QVBoxLayout) -> None:
        """Set up the top bar with dynamic year dropdown, + New Return button, and assessment trigger."""
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Tax Year:"))

        self._year_combo = QComboBox()
        self._year_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._year_combo.setMinimumContentsLength(6)
        self._year_combo.setMinimumWidth(100)
        self._year_combo.currentTextChanged.connect(self._on_year_selected)
        top_bar.addWidget(self._year_combo)

        self._btn_new_return = QPushButton("+ New Return")
        self._btn_new_return.clicked.connect(self._on_new_return_clicked)
        top_bar.addWidget(self._btn_new_return)

        self._btn_assess = QPushButton("Run Initial Assessment")
        self._btn_assess.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: bold; "
            "padding: 5px 12px; border-radius: 4px; }\n"
            "QPushButton:disabled { background-color: #94a3b8; }\n"
            "QPushButton:hover:!disabled { background-color: #1d4ed8; }"
        )
        self._btn_assess.setEnabled(False)

        self._btn_assess.clicked.connect(self._on_run_assessment_clicked)
        top_bar.addWidget(self._btn_assess)

        self._btn_delete_return = QPushButton("🗑️ Delete Return")
        self._btn_delete_return.setEnabled(False)
        self._btn_delete_return.clicked.connect(self._on_delete_return_clicked)
        top_bar.addWidget(self._btn_delete_return)

        top_bar.addStretch()
        root.addLayout(top_bar)

    def _init_form_pane(self) -> None:
        """Set up the left pane for displaying the tax form state."""
        form_pane = QWidget()
        form_layout = QVBoxLayout(form_pane)

        self._form_title_label = QLabel("<b>Tax Return State</b>")
        form_layout.addWidget(self._form_title_label)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Field / Section", "Value", "Status", "Rationale"])
        self._tree.setColumnWidth(0, 180)
        self._tree.setColumnWidth(1, 100)
        self._tree.setColumnWidth(2, 100)
        form_layout.addWidget(self._tree)
        self._splitter.addWidget(form_pane)

    def _init_chat_pane(self) -> None:
        """Set up the right pane for the chat interface."""
        self._chat_pane = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_pane)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._splitter.addWidget(self._chat_pane)

    def _populate_year_combo(self, select_year: int | None = None) -> None:
        """Scan session store and populate tax year dropdown with existing returns."""
        self._year_combo.blockSignals(True)
        self._year_combo.clear()

        existing_years: set[int] = set()
        sessions = self._store.list_sessions()
        for s in sessions:
            if isinstance(s, IrishTaxFilingSession):
                existing_years.add(s.get_tax_year())

        sorted_years = sorted(existing_years, reverse=True)
        for y in sorted_years:
            self._year_combo.addItem(str(y))

        self._year_combo.blockSignals(False)

        target_year = select_year if select_year is not None else (sorted_years[0] if sorted_years else None)
        if target_year is not None:
            idx = self._year_combo.findText(str(target_year))
            if idx >= 0:
                self._year_combo.setCurrentIndex(idx)
                self._on_year_selected(str(target_year))
            else:
                self._year_combo.addItem(str(target_year))
                self._year_combo.setCurrentText(str(target_year))
                self._on_year_selected(str(target_year))
        elif sorted_years:
            self._on_year_selected(str(sorted_years[0]))
        else:
            self._btn_assess.setEnabled(False)
            self._btn_delete_return.setEnabled(False)
            self._form_title_label.setText("<b>Tax Return State: No Active Return</b>")
            self._tree.clear()
            if self._chat_tab is not None:
                self._chat_layout.removeWidget(self._chat_tab)
                self._chat_tab.deleteLater()
                self._chat_tab = None

    def _on_delete_return_clicked(self) -> None:
        """Prompt confirmation and delete the active tax return session to start from scratch."""
        if not self._year_combo.currentText():
            return

        year = self._current_year
        session_id = f"irish_report_{year}"

        reply = QMessageBox.question(
            self,
            "Confirm Delete Return",
            f"Are you sure you want to delete the Irish Tax Return for {year}?\n\n"
            "This will delete the current form draft and chat history for this year so you can start from scratch.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._store.delete_session(session_id)
            if self._chat_tab is not None:
                self._chat_layout.removeWidget(self._chat_tab)
                self._chat_tab.deleteLater()
                self._chat_tab = None

            self._form_state = None
            self._tree.clear()
            self._btn_assess.setEnabled(False)
            self._btn_delete_return.setEnabled(False)
            self._form_title_label.setText("<b>Tax Return State: No Active Return</b>")

            self._populate_year_combo()

    def _on_year_selected(self, year_str: str) -> None:
        """Handle year selection change to immediately load and display return session."""
        if not year_str:
            return
        try:
            year = int(year_str)
        except ValueError:
            return

        self._current_year = year
        session_id = f"irish_report_{self._current_year}"
        session = self._load_or_create_session(session_id)
        self._form_state = session.form_state

        self._setup_chat_tab(session)
        self._refresh_form_view()
        self._poll_timer.start(1000)

    def _on_new_return_clicked(self) -> None:
        """Display dialog to create a new tax return for a selected year."""
        now_year = datetime.now().year
        candidate_years = [str(y) for y in range(now_year, now_year - 6, -1)]

        year_str, ok = QInputDialog.getItem(
            self,
            "New Tax Return",
            "Select Tax Year:",
            candidate_years,
            current=0,
            editable=False,
        )
        if not ok or not year_str:
            return

        selected_year = int(year_str)
        session_id = f"irish_report_{selected_year}"
        existing = self._store.load_session(session_id)

        if existing is not None:
            reply = QMessageBox.question(
                self,
                "Existing Return Found",
                f"A tax return for year {selected_year} already exists.\n\n"
                "Do you want to overwrite it with a fresh return (Yes) or switch to the existing return (No)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                new_session = self._store.create_session(
                    title=f"Tax Return {selected_year}",
                    metadata=TaxFilingMetadata(tax_year=selected_year),
                    form_state=IrishUndeterminedState(tax_year=selected_year),
                    auto_save=False,
                )
                new_session.id = session_id
                self._store.save_session(new_session)
        else:
            new_session = self._store.create_session(
                title=f"Tax Return {selected_year}",
                metadata=TaxFilingMetadata(tax_year=selected_year),
                form_state=IrishUndeterminedState(tax_year=selected_year),
                auto_save=False,
            )
            new_session.id = session_id
            self._store.save_session(new_session)

        self._populate_year_combo(select_year=selected_year)

    def _load_or_create_session(self, session_id: str) -> IrishTaxFilingSession:
        """Load an existing session or create a new one for the tax report."""
        session = self._store.load_session(session_id)
        if session is None:
            session = self._store.create_session(
                title=f"Tax Return {self._current_year}",
                metadata=TaxFilingMetadata(tax_year=self._current_year),
                form_state=IrishUndeterminedState(tax_year=self._current_year),
                auto_save=False,
            )
            session.id = session_id
            self._store.save_session(session)
        return session

    def _setup_chat_tab(self, session: IrishTaxFilingSession) -> None:
        """Initialize and embed the chat tab for the active session."""
        if self._chat_tab is not None:
            self._chat_layout.removeWidget(self._chat_tab)
            self._chat_tab.deleteLater()

        self._chat_tab = ChatTab(db=self._db, session_store=self._store, agent=self._agent)
        self._chat_tab.hide_sidebar()

        # Force load the specific session
        self._chat_tab._active_session = session
        self._chat_tab._render_messages()

        self._chat_layout.addWidget(self._chat_tab)

        self._btn_assess.setEnabled(True)
        self._btn_delete_return.setEnabled(True)
        self._refresh_form_view()

    def _on_run_assessment_clicked(self) -> None:
        """Submit initial data gathering and tax assessment prompt to filing agent."""
        if self._chat_tab is None:
            return
        prompt = (
            f"Please review my financial records, profile, and documents for tax year {self._current_year}. "
            f"Gather all relevant tax facts, evaluate whether I need to file Form 11, Form 12 or Form CG1, "
            f"perform the initial calculations for capital gains / income, and populate the tax form fields."
        )
        self._chat_tab.send_user_message(prompt)

    def _add_section_items(self, title: str, fields_dict: dict[str, FormField]) -> None:
        """Render a form section and its fields in the tree widget."""
        parent = QTreeWidgetItem(self._tree, [title])
        parent.setExpanded(True)
        for key, field in fields_dict.items():
            val = str(field.value) if field.value is not None else ""
            status = field.status
            rationale = field.rationale or ""
            QTreeWidgetItem(parent, [key, val, status, rationale])

    def _refresh_form_view(self) -> None:
        """Refresh the tree widget from the active session form state."""
        if self._chat_tab is not None and isinstance(self._chat_tab._active_session, IrishTaxFilingSession):
            self._form_state = self._chat_tab._active_session.form_state

        if self._form_state is None:
            return

        if self._form_state.form_type == "undetermined":
            form_label = "Undetermined (Assessing Obligations)"
        elif self._form_state.form_type == "form11":
            form_label = "Form 11 (Self-Assessment)"
        elif self._form_state.form_type == "form12":
            form_label = "Form 12 (PAYE Return)"
        else:
            form_label = "Form CG1 (Capital Gains)"

        self._form_title_label.setText(f"<b>Tax Return State: {form_label} ({self._form_state.tax_year})</b>")

        self._tree.clear()

        # Render obligation decision if evaluated
        decision = self._form_state.obligation_decision
        if decision.required_form != "undetermined" or decision.rationale:
            decision_item = QTreeWidgetItem(
                self._tree,
                ["Filing Determination", decision.required_form.upper(), "computed", decision.rationale or ""],
            )
            decision_item.setExpanded(True)

        if (
            isinstance(self._form_state, (IrishForm11State, IrishForm12State, IrishCG1State))
            and self._form_state.capital_gains
        ):
            self._add_section_items("Capital Gains", self._form_state.capital_gains)
        elif isinstance(self._form_state, (IrishForm11State, IrishCG1State)):
            self._add_section_items("Capital Gains", self._form_state.capital_gains)

        if isinstance(self._form_state, (IrishForm11State, IrishForm12State)):
            self._add_section_items("Income", self._form_state.income)

        if isinstance(self._form_state, IrishForm12State):
            self._add_section_items("Tax Credits & Reliefs", self._form_state.tax_credits)

        self._add_section_items("Additional Fields", self._form_state.additional_fields)
