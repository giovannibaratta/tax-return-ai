"""Unit tests for IrishTaxReportTab UI component, UIConfig, Form 12, and filing workflows."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from backend.chat.session_store import SessionStore
from backend.config import AppConfig
from backend.db_manager import DatabaseManager, LocalDb
from src.jurisdiction.ireland.tax_form_models import (
    FormField,
    IrishForm11State,
    IrishForm12State,
    IrishTaxFilingSession,
    IrishUndeterminedState,
    TaxFilingMetadata,
)
from src.ui.config import UIConfig
from src.ui.report_tab import IrishTaxReportTab


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Ensure single QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def db_instance(tmp_path: Path) -> DatabaseManager:
    """Provide isolated DatabaseManager fixture."""
    db_path = str(tmp_path / "test_report_tab.db")
    return DatabaseManager(db_config=LocalDb(db_path=db_path, vector_db_path=str(Path(db_path).parent / "vector.db")))


@pytest.fixture
def ui_config(db_instance: DatabaseManager, tmp_path: Path) -> UIConfig:
    """Provide isolated UIConfig fixture."""
    app_config = AppConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "test_report_tab.db",
        vector_db_path=tmp_path / "vector.db",
        sessions_base_dir=tmp_path / ".sessions",
    )
    return UIConfig(db=db_instance, app_config=app_config)



def test_report_tab_initial_state_empty(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: IrishTaxReportTab initialized with an empty session store
    tab = IrishTaxReportTab(config=ui_config)

    # When: Initialized without pre-existing sessions
    # Then: Dropdown has no items and assessment button is disabled
    assert tab._year_combo.count() == 0
    assert not tab._btn_assess.isEnabled()
    assert tab._btn_new_return.isEnabled()


def test_report_tab_auto_loads_existing_session(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Pre-existing tax filing sessions in store
    store: SessionStore[IrishTaxFilingSession] = SessionStore(
        sessions_dir=ui_config.tax_filing_sessions_dir,
        session_cls=IrishTaxFilingSession,
    )
    s2024 = store.create_session(
        title="Tax Return 2024",
        form_state=IrishUndeterminedState(tax_year=2024),
        auto_save=False,
    )
    s2024.id = "irish_report_2024"
    store.save_session(s2024)

    s2025 = store.create_session(
        title="Tax Return 2025",
        form_state=IrishUndeterminedState(tax_year=2025),
        auto_save=False,
    )
    s2025.id = "irish_report_2025"
    store.save_session(s2025)

    # When: IrishTaxReportTab is initialized
    tab = IrishTaxReportTab(config=ui_config)

    # Then: Dropdown lists years descending and auto-loads the latest year (2025)
    assert tab._year_combo.count() == 2
    assert tab._year_combo.itemText(0) == "2025"
    assert tab._year_combo.itemText(1) == "2024"
    assert tab._year_combo.currentText() == "2025"
    assert tab._btn_assess.isEnabled()
    assert tab._form_state is not None
    assert tab._form_state.tax_year == 2025
    assert tab._chat_tab is not None


def test_report_tab_dropdown_selection_switches_session(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Report tab with 2025 and 2024 sessions
    store: SessionStore[IrishTaxFilingSession] = SessionStore(
        sessions_dir=ui_config.tax_filing_sessions_dir,
        session_cls=IrishTaxFilingSession,
    )
    s2024 = store.create_session(
        title="Tax Return 2024",
        form_state=IrishUndeterminedState(tax_year=2024),
        auto_save=False,
    )
    s2024.id = "irish_report_2024"
    store.save_session(s2024)

    s2025 = store.create_session(
        title="Tax Return 2025",
        form_state=IrishUndeterminedState(tax_year=2025),
        auto_save=False,
    )
    s2025.id = "irish_report_2025"
    store.save_session(s2025)

    tab = IrishTaxReportTab(config=ui_config)

    # When: User selects 2024 from dropdown
    tab._year_combo.setCurrentText("2024")

    # Then: Active form state switches to tax year 2024
    assert tab._form_state is not None
    assert tab._form_state.tax_year == 2024
    assert "2024" in tab._form_title_label.text()


def test_report_tab_new_return_flow(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Report tab without existing sessions
    tab = IrishTaxReportTab(config=ui_config)

    # When: User creates a new return for 2023 via New Return dialog
    with patch("PySide6.QtWidgets.QInputDialog.getItem", return_value=("2023", True)):
        tab._btn_new_return.click()

    # Then: Dropdown selects 2023 and loads new session
    assert tab._year_combo.currentText() == "2023"
    assert tab._form_state is not None
    assert tab._form_state.tax_year == 2023
    assert tab._btn_assess.isEnabled()


def test_report_tab_new_return_existing_overwrite_confirmation(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: An existing 2024 return
    store: SessionStore[IrishTaxFilingSession] = SessionStore(
        sessions_dir=ui_config.tax_filing_sessions_dir,
        session_cls=IrishTaxFilingSession,
    )
    s2024 = store.create_session(
        title="Tax Return 2024",
        form_state=IrishForm11State(tax_year=2024),
        auto_save=False,
    )
    s2024.id = "irish_report_2024"
    store.save_session(s2024)

    tab = IrishTaxReportTab(config=ui_config)
    assert isinstance(tab._form_state, IrishForm11State)

    # When: User clicks New Return for 2024 and confirms overwrite (Yes)
    with (
        patch("PySide6.QtWidgets.QInputDialog.getItem", return_value=("2024", True)),
        patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes),
    ):
        tab._btn_new_return.click()

    # Then: 2024 return is overwritten with fresh IrishUndeterminedState
    assert tab._form_state is not None
    assert tab._form_state.tax_year == 2024
    assert isinstance(tab._form_state, IrishUndeterminedState)


def test_report_tab_run_assessment_trigger(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Active session in report tab
    tab = IrishTaxReportTab(config=ui_config)
    with patch("PySide6.QtWidgets.QInputDialog.getItem", return_value=("2024", True)):
        tab._btn_new_return.click()

    assert tab._chat_tab is not None
    tab._chat_tab.send_user_message = MagicMock()

    # When: User clicks Run Initial Assessment button
    tab._btn_assess.click()

    # Then: Filing assessment prompt sent to agent
    tab._chat_tab.send_user_message.assert_called_once()
    prompt: str = tab._chat_tab.send_user_message.call_args[0][0]
    assert "2024" in prompt
    assert "Form 11" in prompt or "Form 12" in prompt or "Form CG1" in prompt


def test_report_tab_refresh_form_view_form12(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Report tab with Form 12 state populated
    tab = IrishTaxReportTab(config=ui_config)
    with patch("PySide6.QtWidgets.QInputDialog.getItem", return_value=("2024", True)):
        tab._btn_new_return.click()

    form12 = IrishForm12State(tax_year=2024)
    form12.income["paye_employment_income"] = FormField(
        name="paye_employment_income",
        value="65000.00",
        status="computed_via_tool",
        rationale="P60 / Employment Detail Summary",
    )
    form12.tax_credits["rent_tax_credit"] = FormField(
        name="rent_tax_credit",
        value="1000.00",
        status="computed_via_rag",
        rationale="Statutory rent relief claimed",
    )

    if tab._chat_tab and isinstance(tab._chat_tab._active_session, IrishTaxFilingSession):
        tab._chat_tab._active_session.form_state = form12

    # When: Refreshing form view
    tab._refresh_form_view()

    # Then: Tree widget contains Income and Tax Credits sections
    assert "Form 12 (PAYE Return)" in tab._form_title_label.text()
    assert tab._tree.topLevelItemCount() > 0

    section_titles: list[str] = []
    for i in range(tab._tree.topLevelItemCount()):
        item = tab._tree.topLevelItem(i)
        if item is not None:
            section_titles.append(item.text(0))

    assert "Income" in section_titles
    assert "Tax Credits & Reliefs" in section_titles


def test_report_tab_metadata_year_discovery(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Sessions with arbitrary UUIDs and structured metadata
    store: SessionStore[IrishTaxFilingSession] = SessionStore(
        sessions_dir=ui_config.tax_filing_sessions_dir,
        session_cls=IrishTaxFilingSession,
    )
    custom_session = store.create_session(
        title="Custom Filing 2022",
        metadata=TaxFilingMetadata(tax_year=2022),
        form_state=IrishUndeterminedState(tax_year=2022),
        auto_save=False,
    )
    custom_session.id = "session-uuid-12345"
    store.save_session(custom_session)

    # When: Initializing report tab
    tab = IrishTaxReportTab(config=ui_config)

    # Then: Year 2022 is discovered from metadata and populated in combo box
    assert tab._year_combo.count() == 1
    assert tab._year_combo.itemText(0) == "2022"
    assert tab._form_state is not None
    assert tab._form_state.tax_year == 2022


def test_report_tab_delete_return_flow(qapp: QApplication, ui_config: UIConfig) -> None:
    # Given: Report tab with 2025 and 2024 returns
    store: SessionStore[IrishTaxFilingSession] = SessionStore(
        sessions_dir=ui_config.tax_filing_sessions_dir,
        session_cls=IrishTaxFilingSession,
    )
    s2024 = store.create_session(
        title="Tax Return 2024",
        form_state=IrishUndeterminedState(tax_year=2024),
        auto_save=False,
    )
    s2024.id = "irish_report_2024"
    store.save_session(s2024)

    s2025 = store.create_session(
        title="Tax Return 2025",
        form_state=IrishUndeterminedState(tax_year=2025),
        auto_save=False,
    )
    s2025.id = "irish_report_2025"
    store.save_session(s2025)

    tab = IrishTaxReportTab(config=ui_config)
    assert tab._year_combo.currentText() == "2025"
    assert tab._btn_delete_return.isEnabled()

    # When: Deleting 2025 return with confirmation
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
        tab._btn_delete_return.click()

    # Then: 2025 return file is deleted and active year switches to 2024
    assert store.load_session("irish_report_2025") is None
    assert tab._year_combo.count() == 1
    assert tab._year_combo.currentText() == "2024"
    assert tab._form_state is not None
    assert tab._form_state.tax_year == 2024

    # When: Deleting the remaining 2024 return
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
        tab._btn_delete_return.click()

    # Then: Dropdown is empty, buttons disabled, and no active return shown
    assert store.load_session("irish_report_2024") is None
    assert tab._year_combo.count() == 0
    assert not tab._btn_assess.isEnabled()
    assert not tab._btn_delete_return.isEnabled()
    assert "No Active Return" in tab._form_title_label.text()
