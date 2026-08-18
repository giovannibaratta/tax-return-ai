"""Tax Income Records UI tab for Tax Return AI.

Provides a tabbed interface for:
1. Staged Income Documents: Review extracted income records (e.g. Irish EDS),
   inspect 3-voter diffs, edit fields, and approve records into the official ledger.

2. Approved Income Ledger: Browse official verified income records used in tax computations.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backend.db_manager import DatabaseManager
from backend.domain_models import (
    IrishEmploymentDetailSummaryPayload,
    StrictStagedTaxIncomeRecord,
    StrictTaxIncomeRecord,
)
from src.ui.base_tab import BaseAppTab
from src.ui.config import UIConfig
from src.ui.income_details_dialog import IncomeRecordDetailsDialog
from src.ui.income_voter_diff_dialog import IncomeVoterDiffDialog


class IncomeRecordsTab(BaseAppTab):
    """Widget for reviewing staged income documents and browsing approved income records."""

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db

        self._staged_records: list[StrictStagedTaxIncomeRecord] = []
        self._selected_staged_record: StrictStagedTaxIncomeRecord | None = None

        self._ledger_records: list[StrictTaxIncomeRecord] = []
        self._selected_ledger_record: StrictTaxIncomeRecord | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._tabs = QTabWidget(self)

        # Tab 1: Staged Income Documents
        self._staged_tab = QWidget()
        self._init_staged_tab(self._staged_tab)
        self._tabs.addTab(self._staged_tab, "Staged Income Documents")

        # Tab 2: Approved Income Ledger
        self._ledger_tab = QWidget()
        self._init_ledger_tab(self._ledger_tab)
        self._tabs.addTab(self._ledger_tab, "Approved Income Ledger")

        root.addWidget(self._tabs)

        self.load_all_data()

    def reload_config(self, config: UIConfig) -> None:
        """Reload state and database connection from UIConfig."""
        self._db = config.db
        self.load_all_data()

    def load_all_data(self) -> None:
        """Reload both staged records and approved ledger records."""
        self.load_staged_records()
        self.load_ledger_records()

    # -------------------------------------------------------------------------
    # Tab 1: Staged Income Documents
    # -------------------------------------------------------------------------
    def _init_staged_tab(self, parent_widget: QWidget) -> None:
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Controls bar
        top_bar = QHBoxLayout()

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.load_staged_records)
        top_bar.addWidget(btn_refresh)

        top_bar.addWidget(QLabel("Tax Year:"))
        self._combo_staged_year = QComboBox()
        # TODO: We could compute the years dinamically
        self._combo_staged_year.addItems(["All", "2026", "2025", "2024", "2023"])
        self._combo_staged_year.currentIndexChanged.connect(self.load_staged_records)
        top_bar.addWidget(self._combo_staged_year)

        top_bar.addWidget(QLabel("Jurisdiction:"))
        self._combo_staged_jur = QComboBox()
        self._combo_staged_jur.addItems(["All", "Ireland", "Italy"])
        self._combo_staged_jur.currentIndexChanged.connect(self.load_staged_records)
        top_bar.addWidget(self._combo_staged_jur)

        top_bar.addStretch()

        self._btn_diff_voters = QPushButton("🏛️ Inspect / Diff Voters")
        self._btn_diff_voters.setEnabled(False)
        self._btn_diff_voters.setStyleSheet(
            "font-weight: bold; background-color: #2980b9; color: white; padding: 4px 12px;"
        )
        self._btn_diff_voters.clicked.connect(self._open_voter_diff)
        top_bar.addWidget(self._btn_diff_voters)

        self._btn_approve_staged = QPushButton("✅ Approve Selected")
        self._btn_approve_staged.setEnabled(False)
        self._btn_approve_staged.setStyleSheet(
            "font-weight: bold; background-color: #27ae60; color: white; padding: 4px 12px;"
        )
        self._btn_approve_staged.clicked.connect(self._approve_selected_staged)
        top_bar.addWidget(self._btn_approve_staged)

        btn_approve_all_auto = QPushButton("⚡ Approve All Auto-Approved")
        btn_approve_all_auto.clicked.connect(self._approve_all_auto_approved)
        top_bar.addWidget(btn_approve_all_auto)

        self._btn_delete_staged = QPushButton("🗑️ Delete Staged")
        self._btn_delete_staged.setEnabled(False)
        self._btn_delete_staged.clicked.connect(self._delete_selected_staged)
        top_bar.addWidget(self._btn_delete_staged)

        layout.addLayout(top_bar)

        # Staged Table
        self._table_staged = QTableWidget()
        self._table_staged.setColumnCount(12)
        self._table_staged.setHorizontalHeaderLabels(
            [
                "Staged ID",
                "Year",
                "Jurisdiction",
                "Employer Name",
                "Gross Pay",
                "PAYE Tax",
                "USC",
                "PRSI",
                "Source File",
                "Verification Status",
                "Promoted Ledger ID",
                "Discrepancies",
            ]
        )
        self._table_staged.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table_staged.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table_staged.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_staged.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table_staged.itemSelectionChanged.connect(self._on_staged_selection_changed)
        self._table_staged.itemDoubleClicked.connect(self._on_staged_item_double_clicked)
        layout.addWidget(self._table_staged)

    def _on_staged_item_double_clicked(self, _item: QTableWidgetItem) -> None:
        """Handle double-click on a staged record table row."""
        self._open_voter_diff()

    @staticmethod
    def _get_staged_status_display(
        rec: StrictStagedTaxIncomeRecord,
    ) -> tuple[str, str, QColor | None, QColor | None]:
        """Return (status_text, promoted_id_text, text_color, row_bg_color)."""
        is_promoted = rec.approved_tax_income_record_id is not None or rec.verification_status == "approved"
        if is_promoted:
            promoted_id_str = f"#{rec.approved_tax_income_record_id}" if rec.approved_tax_income_record_id else "Yes"
            return "✅ APPROVED", promoted_id_str, QColor("#27ae60"), QColor("#f4fbf7")
        if rec.verification_status == "auto_approved":
            return "⚡ AUTO_APPROVED", "— (Pending)", QColor("#27ae60"), None
        if rec.verification_status == "majority_agreed":
            return "🔵 MAJORITY_AGREED", "— (Pending)", QColor("#2980b9"), None
        if rec.verification_status == "escalated_to_user":
            return "⚠️ ESCALATED", "— (Needs Resolution)", QColor("#d35400"), QColor("#fef9e7")
        return rec.verification_status.upper(), "— (Pending)", None, None

    @classmethod
    def _format_staged_row_items(cls, rec: StrictStagedTaxIncomeRecord) -> list[QTableWidgetItem]:
        payload = rec.payload
        if isinstance(payload, IrishEmploymentDetailSummaryPayload):
            emp_name = payload.employer_name or "N/A"
            gross_str = f"€{payload.gross_pay_eur:,.2f}"
            tax_str = f"€{payload.income_tax_paid_eur:,.2f}"
            usc_str = f"€{payload.usc_paid_eur:,.2f}"
            prsi_str = f"€{payload.prsi_paid_eur:,.2f}"
        else:
            emp_name = "— (Needs Resolution)"
            gross_str = "—"
            tax_str = "—"
            usc_str = "—"
            prsi_str = "—"

        disc_count = str(len(rec.discrepancies)) if rec.discrepancies else "0"
        status_str, promoted_id_str, fg_color, bg_color = cls._get_staged_status_display(rec)

        items = [
            QTableWidgetItem(str(rec.id or 0)),
            QTableWidgetItem(str(rec.tax_year)),
            QTableWidgetItem(rec.jurisdiction.capitalize()),
            QTableWidgetItem(emp_name),
            QTableWidgetItem(gross_str),
            QTableWidgetItem(tax_str),
            QTableWidgetItem(usc_str),
            QTableWidgetItem(prsi_str),
            QTableWidgetItem(rec.source_file_name or "N/A"),
            QTableWidgetItem(status_str),
            QTableWidgetItem(promoted_id_str),
            QTableWidgetItem(disc_count),
        ]

        if fg_color:
            items[9].setForeground(fg_color)
            if status_str.startswith("✅"):
                items[10].setForeground(fg_color)
        if bg_color:
            for it in items:
                it.setBackground(bg_color)

        for item in items:
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return items

    def load_staged_records(self) -> None:
        """Load staged income records from database with active filters."""
        year_str = self._combo_staged_year.currentText()
        tax_year = int(year_str) if year_str != "All" else None

        jur_str = self._combo_staged_jur.currentText().lower()
        jurisdiction = jur_str if jur_str != "all" else None

        self._staged_records = self._db.get_staged_tax_income_records(
            tax_year=tax_year,
            jurisdiction=jurisdiction,
        )

        self._table_staged.setRowCount(len(self._staged_records))
        for row, rec in enumerate(self._staged_records):
            items = self._format_staged_row_items(rec)
            for col, item in enumerate(items):
                self._table_staged.setItem(row, col, item)

        self._on_staged_selection_changed()

    def _on_staged_selection_changed(self) -> None:
        selected_rows = self._table_staged.selectionModel().selectedRows()
        if not selected_rows:
            self._selected_staged_record = None
            self._btn_diff_voters.setEnabled(False)
            self._btn_approve_staged.setEnabled(False)
            self._btn_approve_staged.setText("✅ Approve Selected")
            self._btn_delete_staged.setEnabled(False)
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self._staged_records):
            self._selected_staged_record = self._staged_records[row]
            is_already_approved = (
                self._selected_staged_record.approved_tax_income_record_id is not None
                or self._selected_staged_record.verification_status == "approved"
            )
            self._btn_diff_voters.setEnabled(True)
            self._btn_approve_staged.setEnabled(not is_already_approved)
            self._btn_approve_staged.setText("✅ Already Approved" if is_already_approved else "✅ Approve Selected")
            self._btn_delete_staged.setEnabled(True)

    def _open_voter_diff(self) -> None:
        if not self._selected_staged_record:
            return
        dialog = IncomeVoterDiffDialog(self._selected_staged_record, self._db, self)
        if dialog.exec():
            self.load_all_data()

    def _approve_selected_staged(self) -> None:
        if not self._selected_staged_record or self._selected_staged_record.id is None:
            return
        staged_id = self._selected_staged_record.id
        reply = QMessageBox.question(
            self,
            "Confirm Approval",
            f"Promote staged income record #{staged_id} into the official ledger?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                approved_id = self._db.approve_staged_tax_income_record(staged_id)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Record successfully approved and assigned Ledger ID #{approved_id}!",
                )
                self.load_all_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to approve record: {e}")

    def _approve_all_auto_approved(self) -> None:
        auto_records = [
            r
            for r in self._staged_records
            if r.id is not None and r.verification_status == "auto_approved" and r.approved_tax_income_record_id is None
        ]
        if not auto_records:
            QMessageBox.information(self, "No Records", "No pending auto-approved records to promote.")
            return

        reply = QMessageBox.question(
            self,
            "Approve All Auto-Approved",
            f"Approve and promote all {len(auto_records)} auto-approved record(s) into official ledger?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            promoted_count = 0
            for r in auto_records:
                assert r.id is not None
                try:
                    self._db.approve_staged_tax_income_record(r.id)
                    promoted_count += 1
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed promoting record #{r.id}: {e}")
            QMessageBox.information(self, "Success", f"Promoted {promoted_count} record(s) into the official ledger.")
            self.load_all_data()

    def _delete_selected_staged(self) -> None:
        if not self._selected_staged_record or self._selected_staged_record.id is None:
            return
        staged_id = self._selected_staged_record.id
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete staged record #{staged_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.delete_staged_tax_income_record(staged_id)
            self.load_staged_records()

    # -------------------------------------------------------------------------
    # Tab 2: Approved Income Ledger
    # -------------------------------------------------------------------------
    def _init_ledger_tab(self, parent_widget: QWidget) -> None:
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Controls bar
        top_bar = QHBoxLayout()

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.load_ledger_records)
        top_bar.addWidget(btn_refresh)

        top_bar.addWidget(QLabel("Tax Year:"))
        self._combo_ledger_year = QComboBox()
        self._combo_ledger_year.addItems(["All", "2026", "2025", "2024", "2023"])
        self._combo_ledger_year.currentIndexChanged.connect(self.load_ledger_records)
        top_bar.addWidget(self._combo_ledger_year)

        top_bar.addWidget(QLabel("Jurisdiction:"))
        self._combo_ledger_jur = QComboBox()
        self._combo_ledger_jur.addItems(["All", "Ireland", "Italy"])
        self._combo_ledger_jur.currentIndexChanged.connect(self.load_ledger_records)
        top_bar.addWidget(self._combo_ledger_jur)

        top_bar.addStretch()

        self._btn_view_ledger_details = QPushButton("🔍 View Full Details")
        self._btn_view_ledger_details.setStyleSheet(
            "font-weight: bold; background-color: #2980b9; color: white; padding: 4px 12px;"
        )
        self._btn_view_ledger_details.setEnabled(False)
        self._btn_view_ledger_details.clicked.connect(self._open_ledger_details)
        top_bar.addWidget(self._btn_view_ledger_details)

        self._btn_delete_ledger = QPushButton("🗑️ Delete Record")
        self._btn_delete_ledger.setEnabled(False)
        self._btn_delete_ledger.clicked.connect(self._delete_selected_ledger)
        top_bar.addWidget(self._btn_delete_ledger)

        layout.addLayout(top_bar)

        # Ledger Table
        self._table_ledger = QTableWidget()
        self._table_ledger.setColumnCount(9)
        self._table_ledger.setHorizontalHeaderLabels(
            [
                "Ledger ID",
                "Year",
                "Jurisdiction",
                "Employer Name",
                "Gross Pay",
                "PAYE Tax",
                "USC",
                "PRSI",
                "Created At",
            ]
        )
        self._table_ledger.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table_ledger.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table_ledger.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_ledger.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table_ledger.itemSelectionChanged.connect(self._on_ledger_selection_changed)
        self._table_ledger.itemDoubleClicked.connect(self._open_ledger_details)
        layout.addWidget(self._table_ledger)

    def _open_ledger_details(self, _item: object = None) -> None:
        """Open detailed inspector dialog for selected ledger record."""
        if not self._selected_ledger_record:
            return
        dialog = IncomeRecordDetailsDialog(self._selected_ledger_record, self)
        dialog.exec()

    def load_ledger_records(self) -> None:
        """Load approved income records from database."""
        year_str = self._combo_ledger_year.currentText()
        tax_year = int(year_str) if year_str != "All" else None

        jur_str = self._combo_ledger_jur.currentText().lower()
        jurisdiction = jur_str if jur_str != "all" else None

        self._ledger_records = self._db.get_tax_income_records(
            tax_year=tax_year,
            jurisdiction=jurisdiction,
        )

        self._table_ledger.setRowCount(len(self._ledger_records))
        for row, rec in enumerate(self._ledger_records):
            payload = rec.payload
            emp_name = "N/A"
            gross_str = "N/A"
            tax_str = "N/A"
            usc_str = "N/A"
            prsi_str = "N/A"

            if isinstance(payload, IrishEmploymentDetailSummaryPayload):
                emp_name = payload.employer_name or "N/A"
                gross_str = f"€{payload.gross_pay_eur:,.2f}"
                tax_str = f"€{payload.income_tax_paid_eur:,.2f}"
                usc_str = f"€{payload.usc_paid_eur:,.2f}"
                prsi_str = f"€{payload.prsi_paid_eur:,.2f}"

            created_str = rec.created_at.strftime("%Y-%m-%d %H:%M") if rec.created_at else "N/A"

            items = [
                QTableWidgetItem(str(rec.id or 0)),
                QTableWidgetItem(str(rec.tax_year)),
                QTableWidgetItem(rec.jurisdiction.capitalize()),
                QTableWidgetItem(emp_name),
                QTableWidgetItem(gross_str),
                QTableWidgetItem(tax_str),
                QTableWidgetItem(usc_str),
                QTableWidgetItem(prsi_str),
                QTableWidgetItem(created_str),
            ]

            for col, item in enumerate(items):
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._table_ledger.setItem(row, col, item)

        self._on_ledger_selection_changed()

    def _on_ledger_selection_changed(self) -> None:
        selected_rows = self._table_ledger.selectionModel().selectedRows()
        if not selected_rows:
            self._selected_ledger_record = None
            self._btn_view_ledger_details.setEnabled(False)
            self._btn_delete_ledger.setEnabled(False)
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self._ledger_records):
            self._selected_ledger_record = self._ledger_records[row]
            self._btn_view_ledger_details.setEnabled(True)
            self._btn_delete_ledger.setEnabled(True)

    def _delete_selected_ledger(self) -> None:
        if not self._selected_ledger_record or self._selected_ledger_record.id is None:
            return
        rec_id = self._selected_ledger_record.id
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete approved ledger record #{rec_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.delete_tax_income_record(rec_id)
            self.load_ledger_records()
