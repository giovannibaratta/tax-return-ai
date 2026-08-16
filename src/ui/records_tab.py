"""Financial Records UI tab for Tax Return AI.

Provides a multi-tab interface:
1. Staged Transactions: Review, edit, resolve voter diffs, approve, or reject raw ingested transactions before moving to the ledger.
2. Approved Ledger: Vertical layout for browsing and editing official approved financial records.
3. Portfolio Snapshot: Aggregated portfolio view reconstructing holdings as of current date.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
from backend.db_models import (
    AssetMerger,
    FinancialRecord,
    StagedFinancialRecord,
)
from backend.domain_models import (
    AssetType,
    BaseStrictRecord,
    TransactionAction,
)
from backend.ingestion.openfigi import OpenFIGIMapper
from backend.portfolio import PortfolioEngine, PortfolioFilter, PortfolioPosition, PortfolioSnapshot


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


class FinancialRecordsTab(QWidget):
    """Widget for reviewing staged transactions, browsing approved records, and viewing portfolio snapshot."""

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._figi_mapper = OpenFIGIMapper()

        self._staged_records: list[StagedFinancialRecord] = []
        self._selected_staged_record: StagedFinancialRecord | None = None

        self._ledger_records: list[FinancialRecord] = []
        self._selected_ledger_record: FinancialRecord | None = None

        self._portfolio_snapshot: PortfolioSnapshot | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._tabs = QTabWidget(self)

        # Tab 1: Staged Transactions (Approval / Review Workflow)
        self._staged_tab = QWidget()
        self._init_staged_tab(self._staged_tab)
        self._tabs.addTab(self._staged_tab, "Staged Transactions")

        # Tab 2: Approved Ledger
        self._ledger_tab = QWidget()
        self._init_ledger_tab(self._ledger_tab)
        self._tabs.addTab(self._ledger_tab, "Approved Ledger")

        # Tab 3: Portfolio Snapshot
        self._portfolio_tab = QWidget()
        self._init_portfolio_tab(self._portfolio_tab)
        self._tabs.addTab(self._portfolio_tab, "Portfolio Snapshot")

        # Tab 4: ETF Mergers & Corporate Actions
        self._mergers_tab = QWidget()
        self._init_mergers_tab(self._mergers_tab)
        self._tabs.addTab(self._mergers_tab, "ETF Mergers")

        root.addWidget(self._tabs)

        # Load initial data
        self.load_all_data()

    def load_records(self) -> None:
        """Backward-compatible alias for loading all records."""
        self.load_all_data()

    @property
    def _table(self) -> QTableWidget:
        """Backward-compatible table property pointing to official ledger table."""
        return self._table_ledger

    def _delete_record(self) -> None:
        """Backward-compatible alias for deleting selected ledger record."""
        self._delete_ledger_record()

    def _approve_record(self) -> None:
        """Backward-compatible alias for approving record."""
        if self._selected_staged_record:
            self._approve_staged_record()
        elif self._selected_ledger_record:
            reply = QMessageBox.question(
                self,
                "Confirm Approval",
                f"Are you sure you want to approve record #{self._selected_ledger_record.id}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._selected_ledger_record.verification_status = "approved"
                strict_rec = BaseStrictRecord.from_raw(self._selected_ledger_record)
                self._db.update_strict_financial_record(strict_rec)
                self.load_ledger_records()

    def _flag_attention(self) -> None:
        """Backward-compatible alias for flagging record for attention."""
        if self._selected_ledger_record:
            reply = QMessageBox.question(
                self,
                "Confirm Flag for Attention",
                f"Are you sure you want to flag record #{self._selected_ledger_record.id} for user attention (escalate status)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._selected_ledger_record.verification_status = "escalated_to_user"
                strict_rec = BaseStrictRecord.from_raw(self._selected_ledger_record)
                self._db.update_strict_financial_record(strict_rec)
                self.load_ledger_records()

    def load_all_data(self) -> None:
        """Reload all tabs."""
        self.load_staged_records()
        self.load_ledger_records()
        self.load_portfolio_snapshot()
        self.load_asset_mergers()

    # ------------------------------------------------------------------
    # Tab 1: Staged Transactions (Approval Pipeline)
    # ------------------------------------------------------------------

    def _init_staged_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Filter bar
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("Account Country:"))
        self._staged_filter_account_country = QComboBox()
        self._staged_filter_account_country.addItems(["All", "italy", "ireland"])
        self._staged_filter_account_country.currentTextChanged.connect(self.load_staged_records)
        bar.addWidget(self._staged_filter_account_country)

        bar.addWidget(QLabel("Status:"))
        self._staged_filter_status = QComboBox()
        self._staged_filter_status.addItems(["All", "pending_approval", "escalated_to_user", "approved", "rejected"])
        self._staged_filter_status.currentTextChanged.connect(self.load_staged_records)
        bar.addWidget(self._staged_filter_status)

        bar.addWidget(QLabel("Search:"))
        self._staged_filter_search = QLineEdit()
        self._staged_filter_search.setPlaceholderText("Symbol, provider, ISIN...")
        self._staged_filter_search.textChanged.connect(self.load_staged_records)
        bar.addWidget(self._staged_filter_search)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.load_staged_records)
        bar.addWidget(btn_refresh)

        btn_backfill = QPushButton("⚡ Backfill OpenFIGI Tickers")
        btn_backfill.setToolTip("Query OpenFIGI to resolve missing short tickers for all staged records with ISINs")
        btn_backfill.clicked.connect(self._backfill_openfigi_staged)
        bar.addWidget(btn_backfill)

        bar.addStretch()
        layout.addLayout(bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical, tab)
        splitter.setHandleWidth(4)

        # Table section
        table_container = QWidget()
        t_layout = QVBoxLayout(table_container)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(4)
        t_layout.addWidget(_section_label("Staged Ingested Transactions (Awaiting Approval / Review)"))

        self._table_staged = QTableWidget(0, 14)
        headers = [
            "Status",
            "ID",
            "Date",
            "Provider",
            "Account Country",
            "Tax Year",
            "Asset Type",
            "Symbol / Name",
            "Action",
            "Qty",
            "Unit Price",
            "Total",
            "Local Total (EUR)",
            "Approvable?",
        ]
        self._table_staged.setHorizontalHeaderLabels(headers)
        self._table_staged.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_staged.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_staged.setAlternatingRowColors(True)
        self._table_staged.itemSelectionChanged.connect(self._on_staged_selection_changed)

        h = self._table_staged.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)

        t_layout.addWidget(self._table_staged)
        splitter.addWidget(table_container)

        # Detail & Edit Section
        detail_container = QWidget()
        d_layout = QVBoxLayout(detail_container)
        d_layout.setContentsMargins(0, 4, 0, 0)
        d_layout.setSpacing(6)

        d_layout.addWidget(_section_label("Staged Record Details & Action"))

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Record ID:"), 0, 0)
        self._staged_edit_id = QLineEdit()
        self._staged_edit_id.setReadOnly(True)
        self._staged_edit_id.setMaximumWidth(80)
        grid.addWidget(self._staged_edit_id, 0, 1)

        grid.addWidget(QLabel("Provider:"), 0, 2)
        self._staged_edit_provider = QLineEdit()
        grid.addWidget(self._staged_edit_provider, 0, 3)

        grid.addWidget(QLabel("Account Country:"), 0, 4)
        self._staged_edit_account_country = QComboBox()
        self._staged_edit_account_country.addItems(["italy", "ireland", "other"])
        grid.addWidget(self._staged_edit_account_country, 0, 5)

        grid.addWidget(QLabel("Tax Year:"), 0, 6)
        self._staged_edit_tax_year = QLineEdit()
        grid.addWidget(self._staged_edit_tax_year, 0, 7)

        grid.addWidget(QLabel("Event Date:"), 1, 0)
        self._staged_edit_timestamp = QLineEdit()
        grid.addWidget(self._staged_edit_timestamp, 1, 1)

        grid.addWidget(QLabel("Asset Type:"), 1, 2)
        self._staged_edit_asset_type = QComboBox()
        self._staged_edit_asset_type.addItems([e.value for e in AssetType])
        grid.addWidget(self._staged_edit_asset_type, 1, 3)

        grid.addWidget(QLabel("Action:"), 1, 4)
        self._staged_edit_action = QComboBox()
        self._staged_edit_action.addItems([e.value for e in TransactionAction])
        grid.addWidget(self._staged_edit_action, 1, 5)

        grid.addWidget(QLabel("Symbol:"), 1, 6)
        self._staged_edit_symbol = QLineEdit()
        grid.addWidget(self._staged_edit_symbol, 1, 7)

        grid.addWidget(QLabel("ISIN:"), 2, 0)
        self._staged_edit_isin = QLineEdit()
        grid.addWidget(self._staged_edit_isin, 2, 1)

        grid.addWidget(QLabel("Asset Name:"), 2, 2)
        self._staged_edit_asset_name = QLineEdit()
        grid.addWidget(self._staged_edit_asset_name, 2, 3, 1, 3)

        grid.addWidget(QLabel("Status:"), 2, 5)
        self._staged_edit_status = QComboBox()
        self._staged_edit_status.addItems(["pending_approval", "escalated_to_user", "approved", "rejected"])
        grid.addWidget(self._staged_edit_status, 2, 6, 1, 2)

        # OpenFIGI Detection Row
        grid.addWidget(QLabel("OpenFIGI Detected:"), 3, 0)
        self._staged_edit_openfigi_detected = QLineEdit()
        self._staged_edit_openfigi_detected.setReadOnly(True)
        grid.addWidget(self._staged_edit_openfigi_detected, 3, 1)

        figi_btn_box = QHBoxLayout()
        btn_apply_figi = QPushButton("← Use Detected Ticker")
        btn_apply_figi.setToolTip("Copy OpenFIGI detected ticker into Symbol input")
        btn_apply_figi.clicked.connect(self._apply_staged_openfigi_symbol)
        figi_btn_box.addWidget(btn_apply_figi)

        btn_fetch_figi = QPushButton("🔍 Fetch OpenFIGI")
        btn_fetch_figi.setToolTip("Query OpenFIGI API to resolve ISIN")
        btn_fetch_figi.clicked.connect(self._fetch_openfigi_for_staged_record)
        figi_btn_box.addWidget(btn_fetch_figi)
        figi_btn_box.addStretch()
        grid.addLayout(figi_btn_box, 3, 2, 1, 2)

        grid.addWidget(QLabel("Quantity:"), 3, 4)
        self._staged_edit_quantity = QLineEdit()
        grid.addWidget(self._staged_edit_quantity, 3, 5)

        grid.addWidget(QLabel("Unit Price:"), 3, 6)
        self._staged_edit_unit_price = QLineEdit()
        grid.addWidget(self._staged_edit_unit_price, 3, 7)

        grid.addWidget(QLabel("Currency:"), 4, 0)
        self._staged_edit_currency = QLineEdit()
        grid.addWidget(self._staged_edit_currency, 4, 1)

        grid.addWidget(QLabel("Fees:"), 4, 2)
        self._staged_edit_fees = QLineEdit()
        grid.addWidget(self._staged_edit_fees, 4, 3)

        grid.addWidget(QLabel("Total Amount:"), 4, 4)
        self._staged_edit_total_amount = QLineEdit()
        grid.addWidget(self._staged_edit_total_amount, 4, 5)

        grid.addWidget(QLabel("FX Rate (EUR):"), 4, 6)
        self._staged_edit_fx_rate = QLineEdit()
        grid.addWidget(self._staged_edit_fx_rate, 4, 7)

        grid.addWidget(QLabel("Local Total:"), 5, 0)
        self._staged_edit_local_total = QLineEdit()
        grid.addWidget(self._staged_edit_local_total, 5, 1)

        grid.addWidget(QLabel("Source File:"), 5, 2)
        self._staged_edit_source_file = QLineEdit()
        self._staged_edit_source_file.setReadOnly(True)
        grid.addWidget(self._staged_edit_source_file, 5, 3, 1, 5)

        d_layout.addLayout(grid)

        d_layout.addWidget(QLabel("Consensus Log & Audit Notes:"))
        self._staged_edit_consensus_log = QTextEdit()
        self._staged_edit_consensus_log.setFont(QFont("Menlo", 10))
        self._staged_edit_consensus_log.setMaximumHeight(65)
        d_layout.addWidget(self._staged_edit_consensus_log)

        # Action Buttons
        btn_row = QHBoxLayout()

        self._btn_staged_save = QPushButton("Save Staged Changes")
        self._btn_staged_save.clicked.connect(self._save_staged_changes)
        btn_row.addWidget(self._btn_staged_save)

        self._btn_staged_compare_voters = QPushButton("🏛️ Compare Voters / Merge Diff")
        self._btn_staged_compare_voters.clicked.connect(self._open_staged_voter_diff_dialog)
        btn_row.addWidget(self._btn_staged_compare_voters)

        self._btn_staged_approve = QPushButton("✅ Approve into Ledger")
        self._btn_staged_approve.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 5px 12px;"
        )
        self._btn_staged_approve.clicked.connect(self._approve_staged_record)
        btn_row.addWidget(self._btn_staged_approve)

        self._btn_staged_reject = QPushButton("❌ Reject Record")
        self._btn_staged_reject.setStyleSheet("background-color: #c0392b; color: white; padding: 5px 12px;")
        self._btn_staged_reject.clicked.connect(self._reject_staged_record)
        btn_row.addWidget(self._btn_staged_reject)

        self._btn_staged_delete = QPushButton("Delete Staged Record")
        self._btn_staged_delete.clicked.connect(self._delete_staged_record)
        btn_row.addWidget(self._btn_staged_delete)

        btn_row.addStretch()
        d_layout.addLayout(btn_row)

        splitter.addWidget(detail_container)
        splitter.setSizes([380, 320])
        layout.addWidget(splitter)

    def load_staged_records(self) -> None:
        """Fetch staged records from DB and apply UI filters."""
        jur_filter = self._staged_filter_account_country.currentText()
        jur = None if jur_filter == "All" else jur_filter

        status_filter = self._staged_filter_status.currentText()
        status = None if status_filter == "All" else status_filter

        search_query = self._staged_filter_search.text().strip().lower()

        try:
            records = self._db.get_staged_records(
                account_country=jur,
                verification_status=status,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Database Error", f"Failed to load staged records:\n{exc}")
            return

        if search_query:
            filtered: list[StagedFinancialRecord] = []
            for r in records:
                text_blob = f"{r.symbol or ''} {r.provider or ''} {r.isin or ''} {r.asset_name or ''} {r.openfigi_detected or ''}".lower()
                if search_query in text_blob:
                    filtered.append(r)
            records = filtered

        self._staged_records = records
        self._populate_staged_table()

    def _populate_staged_table(self) -> None:
        self._table_staged.setRowCount(0)
        for record in self._staged_records:
            row = self._table_staged.rowCount()
            self._table_staged.insertRow(row)

            # Column 0: Status
            st = record.verification_status or "pending_approval"
            if st == "escalated_to_user":
                status_item = QTableWidgetItem("⚠️ Escalated")
                status_item.setBackground(QColor(255, 230, 230))
                status_item.setForeground(QColor(180, 0, 0))
            elif st == "approved":
                status_item = QTableWidgetItem("✅ Approved")
                status_item.setBackground(QColor(230, 255, 230))
                status_item.setForeground(QColor(0, 120, 0))
            elif st == "rejected":
                status_item = QTableWidgetItem("❌ Rejected")
                status_item.setBackground(QColor(245, 245, 245))
                status_item.setForeground(QColor(120, 120, 120))
            else:
                status_item = QTableWidgetItem("⏳ Pending")
                status_item.setBackground(QColor(255, 255, 220))
                status_item.setForeground(QColor(140, 100, 0))
            self._table_staged.setItem(row, 0, status_item)

            self._table_staged.setItem(row, 1, QTableWidgetItem(str(record.id)))

            ts_str = record.event_timestamp.strftime("%Y-%m-%d %H:%M") if record.event_timestamp else "-"
            self._table_staged.setItem(row, 2, QTableWidgetItem(ts_str))
            self._table_staged.setItem(row, 3, QTableWidgetItem(record.provider or "-"))
            self._table_staged.setItem(row, 4, QTableWidgetItem(record.account_country or "-"))
            self._table_staged.setItem(row, 5, QTableWidgetItem(str(record.tax_year) if record.tax_year else "-"))
            self._table_staged.setItem(row, 6, QTableWidgetItem(record.asset_type or "-"))

            sym_name = record.symbol or record.openfigi_detected or record.asset_name or record.isin or "-"
            self._table_staged.setItem(row, 7, QTableWidgetItem(sym_name))
            self._table_staged.setItem(row, 8, QTableWidgetItem(record.action or "-"))

            qty_str = f"{record.quantity:.4f}" if record.quantity is not None else "-"
            self._table_staged.setItem(row, 9, QTableWidgetItem(qty_str))

            price_str = f"{record.unit_price:.2f}" if record.unit_price is not None else "-"
            self._table_staged.setItem(row, 10, QTableWidgetItem(price_str))

            curr = record.currency or "EUR"
            tot_str = f"{record.total_amount:.2f} {curr}" if record.total_amount is not None else "-"
            self._table_staged.setItem(row, 11, QTableWidgetItem(tot_str))

            loc_tot = record.local_total_amount
            loc_str = f"€{loc_tot:.2f}" if loc_tot is not None else "-"
            self._table_staged.setItem(row, 12, QTableWidgetItem(loc_str))

            is_app = record.is_approvable()
            app_item = QTableWidgetItem("Yes" if is_app else "No")
            if not is_app:
                app_item.setForeground(QColor(180, 0, 0))
            self._table_staged.setItem(row, 13, app_item)

            status_item.setData(Qt.ItemDataRole.UserRole, record)

    def _on_staged_selection_changed(self) -> None:
        selected_rows = self._table_staged.selectedItems()
        if not selected_rows:
            self._selected_staged_record = None
            self._clear_staged_form()
            return

        row = self._table_staged.currentRow()
        item = self._table_staged.item(row, 0)
        if item is not None:
            record: StagedFinancialRecord = item.data(Qt.ItemDataRole.UserRole)
            self._selected_staged_record = record
            self._populate_staged_form(record)

    def _populate_staged_form(self, record: StagedFinancialRecord) -> None:
        self._staged_edit_id.setText(str(record.id or ""))
        self._staged_edit_provider.setText(record.provider or "")
        idx_jur = self._staged_edit_account_country.findText(record.account_country or "italy")
        if idx_jur >= 0:
            self._staged_edit_account_country.setCurrentIndex(idx_jur)

        self._staged_edit_tax_year.setText(str(record.tax_year or ""))
        self._staged_edit_timestamp.setText(record.event_timestamp.isoformat() if record.event_timestamp else "")

        idx_asset = self._staged_edit_asset_type.findText(record.asset_type or "stock")
        if idx_asset >= 0:
            self._staged_edit_asset_type.setCurrentIndex(idx_asset)

        idx_action = self._staged_edit_action.findText(record.action or "buy")
        if idx_action >= 0:
            self._staged_edit_action.setCurrentIndex(idx_action)

        self._staged_edit_symbol.setText(record.symbol or "")
        self._staged_edit_isin.setText(record.isin or "")
        self._staged_edit_asset_name.setText(record.asset_name or "")
        self._staged_edit_openfigi_detected.setText(record.openfigi_detected or "")

        idx_st = self._staged_edit_status.findText(record.verification_status or "pending_approval")
        if idx_st >= 0:
            self._staged_edit_status.setCurrentIndex(idx_st)

        self._staged_edit_quantity.setText(str(record.quantity) if record.quantity is not None else "")
        self._staged_edit_unit_price.setText(str(record.unit_price) if record.unit_price is not None else "")
        self._staged_edit_currency.setText(record.currency or "EUR")
        self._staged_edit_fees.setText(str(record.fees) if record.fees is not None else "")
        self._staged_edit_total_amount.setText(str(record.total_amount) if record.total_amount is not None else "")
        self._staged_edit_fx_rate.setText(str(record.fx_rate) if record.fx_rate is not None else "")

        loc_tot = record.local_total_amount
        if loc_tot is None and record.total_amount is not None:
            if (record.currency or "EUR").upper() == "EUR":
                loc_tot = record.total_amount
            elif record.fx_rate is not None:
                loc_tot = record.total_amount * record.fx_rate
        self._staged_edit_local_total.setText(str(loc_tot) if loc_tot is not None else "")

        self._staged_edit_source_file.setText(record.source_file_name or "")

        consensus_str = record.consensus_log or ""
        if consensus_str:
            try:
                parsed = json.loads(consensus_str)
                formatted = json.dumps(parsed, indent=2)
                self._staged_edit_consensus_log.setPlainText(formatted)
            except Exception:
                self._staged_edit_consensus_log.setPlainText(consensus_str)
        else:
            self._staged_edit_consensus_log.setPlainText("(No consensus logs available)")

    def _clear_staged_form(self) -> None:
        self._staged_edit_id.clear()
        self._staged_edit_provider.clear()
        self._staged_edit_tax_year.clear()
        self._staged_edit_timestamp.clear()
        self._staged_edit_symbol.clear()
        self._staged_edit_isin.clear()
        self._staged_edit_asset_name.clear()
        self._staged_edit_openfigi_detected.clear()
        self._staged_edit_quantity.clear()
        self._staged_edit_unit_price.clear()
        self._staged_edit_currency.setText("EUR")
        self._staged_edit_fees.clear()
        self._staged_edit_total_amount.clear()
        self._staged_edit_fx_rate.clear()
        self._staged_edit_local_total.clear()
        self._staged_edit_source_file.clear()
        self._staged_edit_consensus_log.clear()

    def _apply_staged_openfigi_symbol(self) -> None:
        figi_sym = self._staged_edit_openfigi_detected.text().strip()
        if figi_sym:
            self._staged_edit_symbol.setText(figi_sym)

    def _fetch_openfigi_for_staged_record(self) -> None:
        if not self._selected_staged_record:
            QMessageBox.warning(self, "No Selection", "Please select a staged record first.")
            return

        isin = self._staged_edit_isin.text().strip().upper()
        if not isin or len(isin) != 12:
            QMessageBox.warning(self, "Invalid ISIN", "Please enter a valid 12-character ISIN code.")
            return

        figi_res = self._figi_mapper.map_isin(isin)
        ticker, name = figi_res.ticker, figi_res.name
        if ticker:
            self._staged_edit_openfigi_detected.setText(ticker)
            if not self._staged_edit_symbol.text().strip():
                self._staged_edit_symbol.setText(ticker)
            if name and not self._staged_edit_asset_name.text().strip():
                self._staged_edit_asset_name.setText(name)

            self._selected_staged_record.openfigi_detected = ticker
            self._selected_staged_record.symbol = self._staged_edit_symbol.text().strip()
            self._selected_staged_record.asset_name = self._staged_edit_asset_name.text().strip()
            self._db.update_staged_record(self._selected_staged_record)
            self.load_staged_records()
            QMessageBox.information(
                self, "OpenFIGI Found", f"Resolved OpenFIGI Ticker: {ticker}\nCompany Name: {name or 'N/A'}"
            )
        else:
            QMessageBox.warning(self, "Not Found", f"OpenFIGI could not resolve ISIN: {isin}")

    def _backfill_openfigi_staged(self) -> None:
        staged_records = self._db.get_staged_records()
        count = 0
        for r in staged_records:
            if r.isin and (not r.symbol or not r.openfigi_detected):
                figi_res = self._figi_mapper.map_isin(r.isin)
                ticker, name = figi_res.ticker, figi_res.name
                if ticker:
                    r.openfigi_detected = ticker
                    if not r.symbol:
                        r.symbol = ticker
                    if name and not r.asset_name:
                        r.asset_name = name
                    self._db.update_staged_record(r)
                    count += 1

        self.load_staged_records()
        QMessageBox.information(self, "Backfill Complete", f"Backfilled OpenFIGI tickers for {count} staged record(s).")

    def _save_staged_changes(self) -> None:
        if not self._selected_staged_record:
            QMessageBox.warning(self, "No Selection", "Please select a staged record from the table to edit.")
            return

        rec = self._selected_staged_record
        rec.provider = self._staged_edit_provider.text().strip() or None
        rec.account_country = self._staged_edit_account_country.currentText()
        rec.tax_year = self._parse_int(self._staged_edit_tax_year.text())
        rec.event_timestamp = self._parse_datetime(self._staged_edit_timestamp.text()) or rec.event_timestamp
        rec.asset_type = self._staged_edit_asset_type.currentText()
        rec.action = self._staged_edit_action.currentText()
        rec.symbol = self._staged_edit_symbol.text().strip() or None
        rec.isin = self._staged_edit_isin.text().strip() or None
        rec.asset_name = self._staged_edit_asset_name.text().strip() or None
        rec.openfigi_detected = self._staged_edit_openfigi_detected.text().strip() or None
        rec.quantity = self._parse_decimal(self._staged_edit_quantity.text())
        rec.unit_price = self._parse_decimal(self._staged_edit_unit_price.text())
        rec.currency = self._staged_edit_currency.text().strip().upper() or "EUR"
        rec.fees = self._parse_decimal(self._staged_edit_fees.text())
        rec.total_amount = self._parse_decimal(self._staged_edit_total_amount.text())
        rec.fx_rate = self._parse_decimal(self._staged_edit_fx_rate.text())
        rec.local_total_amount = self._parse_decimal(self._staged_edit_local_total.text())

        if rec.local_total_amount is None and rec.total_amount is not None:
            if rec.currency == "EUR":
                rec.local_total_amount = rec.total_amount
            elif rec.fx_rate is not None:
                rec.local_total_amount = rec.total_amount * rec.fx_rate

        rec.verification_status = self._staged_edit_status.currentText()

        try:
            updated = self._db.update_staged_record(rec)
            self._selected_staged_record = updated
            QMessageBox.information(self, "Saved", f"Staged record #{updated.id} updated.")
            self.load_staged_records()
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save staged changes:\n{exc}")

    def _approve_staged_record(self) -> None:
        if not self._selected_staged_record:
            QMessageBox.warning(self, "No Selection", "Please select a staged record to approve.")
            return

        rec = self._selected_staged_record
        if rec.id is None:
            return

        success, msg, created_rec, duplicates = self._db.approve_staged_record(rec.id, force_duplicate=False)

        if not success and msg == "potential_duplicate":
            dup_details: list[str] = []
            for d in duplicates:
                dup_details.append(
                    f"• ID #{d.id}: {d.event_timestamp.strftime('%Y-%m-%d') if d.event_timestamp else 'N/A'} | "
                    f"{(d.action or '').upper()} {d.quantity} {d.symbol or d.asset_type} @ {d.unit_price} {d.currency} ({d.provider})"
                )
            details_str = "\n".join(dup_details)

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("⚠️ Potential Duplicate Detected")
            box.setText(
                f"A matching transaction already exists in the approved ledger:\n\n{details_str}\n\n"
                f"Do you want to approve this staged transaction anyway (creating a new record) or keep it in staging?"
            )
            btn_force = box.addButton("Approve as New Record", QMessageBox.ButtonRole.AcceptRole)
            btn_keep = box.addButton("Keep in Staging", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(btn_keep)

            box.exec()

            if box.clickedButton() == btn_force:
                success, msg, created_rec, _ = self._db.approve_staged_record(rec.id, force_duplicate=True)

        if success:
            QMessageBox.information(
                self,
                "Record Approved",
                f"✅ Staged record #{rec.id} approved and moved to Official Ledger (Record ID #{created_rec.id if created_rec else 'N/A'}).",
            )
            self.load_all_data()
        else:
            QMessageBox.warning(self, "Approval Blocked", f"Cannot approve record:\n\n{msg}")

    def _reject_staged_record(self) -> None:
        if not self._selected_staged_record or self._selected_staged_record.id is None:
            return
        rec_id = self._selected_staged_record.id
        reply = QMessageBox.question(
            self,
            "Confirm Rejection",
            f"Are you sure you want to reject staged record #{rec_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.reject_staged_record(rec_id)
            QMessageBox.information(self, "Record Rejected", f"Record #{rec_id} status set to rejected.")
            self.load_staged_records()

    def _delete_staged_record(self) -> None:
        if not self._selected_staged_record or self._selected_staged_record.id is None:
            return
        rec_id = self._selected_staged_record.id
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete staged record #{rec_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.delete_staged_record_by_id(rec_id)
            self._selected_staged_record = None
            self._clear_staged_form()
            self.load_staged_records()

    def _open_staged_voter_diff_dialog(self) -> None:
        if not self._selected_staged_record:
            QMessageBox.warning(self, "No Selection", "Please select a staged record to compare voters.")
            return

        from PySide6.QtWidgets import QDialog

        from src.ui.voter_diff_dialog import VoterDiffMergeDialog

        dialog = VoterDiffMergeDialog(self._selected_staged_record, self._db, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.load_staged_records()

    # ------------------------------------------------------------------
    # Tab 2: Approved Ledger
    # ------------------------------------------------------------------

    def _init_ledger_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("Account Country:"))
        self._ledger_filter_account_country = QComboBox()
        self._ledger_filter_account_country.addItems(["All", "italy", "ireland"])
        self._ledger_filter_account_country.currentTextChanged.connect(self.load_ledger_records)
        bar.addWidget(self._ledger_filter_account_country)

        bar.addWidget(QLabel("Search:"))
        self._ledger_filter_search = QLineEdit()
        self._ledger_filter_search.setPlaceholderText("Symbol, provider, ISIN...")
        self._ledger_filter_search.textChanged.connect(self.load_ledger_records)
        bar.addWidget(self._ledger_filter_search)

        btn_refresh = QPushButton("Refresh Ledger")
        btn_refresh.clicked.connect(self.load_ledger_records)
        bar.addWidget(btn_refresh)

        btn_backfill_ledger = QPushButton("⚡ Backfill OpenFIGI Tickers")
        btn_backfill_ledger.setToolTip(
            "Query OpenFIGI to resolve missing short tickers for all ledger records with ISINs"
        )
        btn_backfill_ledger.clicked.connect(self._backfill_openfigi_ledger)
        bar.addWidget(btn_backfill_ledger)

        bar.addStretch()
        layout.addLayout(bar)

        splitter = QSplitter(Qt.Orientation.Vertical, tab)
        splitter.setHandleWidth(4)

        t_container = QWidget()
        t_layout = QVBoxLayout(t_container)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(4)
        t_layout.addWidget(_section_label("Official Approved Ledger (financial_records)"))

        self._table_ledger = QTableWidget(0, 14)
        headers = [
            "Status",
            "ID",
            "Date",
            "Provider",
            "Account Country",
            "Tax Year",
            "Asset Type",
            "Symbol / Name",
            "Action",
            "Qty",
            "Unit Price",
            "Total",
            "Local Total (EUR)",
            "Strict Valid?",
        ]
        self._table_ledger.setHorizontalHeaderLabels(headers)
        self._table_ledger.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_ledger.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_ledger.setAlternatingRowColors(True)
        self._table_ledger.itemSelectionChanged.connect(self._on_ledger_selection_changed)

        h = self._table_ledger.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)

        t_layout.addWidget(self._table_ledger)
        splitter.addWidget(t_container)

        d_container = QWidget()
        d_layout = QVBoxLayout(d_container)
        d_layout.setContentsMargins(0, 4, 0, 0)
        d_layout.setSpacing(6)
        d_layout.addWidget(_section_label("Approved Record Inspector"))

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Record ID:"), 0, 0)
        self._ledger_edit_id = QLineEdit()
        self._ledger_edit_id.setReadOnly(True)
        self._ledger_edit_id.setMaximumWidth(80)
        grid.addWidget(self._ledger_edit_id, 0, 1)

        grid.addWidget(QLabel("Provider:"), 0, 2)
        self._ledger_edit_provider = QLineEdit()
        grid.addWidget(self._ledger_edit_provider, 0, 3)

        grid.addWidget(QLabel("Account Country:"), 0, 4)
        self._ledger_edit_account_country = QComboBox()
        self._ledger_edit_account_country.addItems(["italy", "ireland", "other"])
        grid.addWidget(self._ledger_edit_account_country, 0, 5)

        grid.addWidget(QLabel("Tax Year:"), 0, 6)
        self._ledger_edit_tax_year = QLineEdit()
        grid.addWidget(self._ledger_edit_tax_year, 0, 7)

        grid.addWidget(QLabel("Event Date:"), 1, 0)
        self._ledger_edit_timestamp = QLineEdit()
        grid.addWidget(self._ledger_edit_timestamp, 1, 1)

        grid.addWidget(QLabel("Asset Type:"), 1, 2)
        self._ledger_edit_asset_type = QComboBox()
        self._ledger_edit_asset_type.addItems([e.value for e in AssetType])
        grid.addWidget(self._ledger_edit_asset_type, 1, 3)

        grid.addWidget(QLabel("Action:"), 1, 4)
        self._ledger_edit_action = QComboBox()
        self._ledger_edit_action.addItems([e.value for e in TransactionAction])
        grid.addWidget(self._ledger_edit_action, 1, 5)

        grid.addWidget(QLabel("Symbol:"), 1, 6)
        self._ledger_edit_symbol = QLineEdit()
        grid.addWidget(self._ledger_edit_symbol, 1, 7)

        grid.addWidget(QLabel("ISIN:"), 2, 0)
        self._ledger_edit_isin = QLineEdit()
        grid.addWidget(self._ledger_edit_isin, 2, 1)

        grid.addWidget(QLabel("Asset Name:"), 2, 2)
        self._ledger_edit_asset_name = QLineEdit()
        grid.addWidget(self._ledger_edit_asset_name, 2, 3, 1, 3)

        # OpenFIGI Detection Row
        grid.addWidget(QLabel("OpenFIGI Detected:"), 3, 0)
        self._ledger_edit_openfigi_detected = QLineEdit()
        self._ledger_edit_openfigi_detected.setReadOnly(True)
        grid.addWidget(self._ledger_edit_openfigi_detected, 3, 1)

        figi_btn_box_l = QHBoxLayout()
        btn_apply_figi_l = QPushButton("← Use Detected Ticker")
        btn_apply_figi_l.setToolTip("Copy OpenFIGI detected ticker into Symbol input")
        btn_apply_figi_l.clicked.connect(self._apply_ledger_openfigi_symbol)
        figi_btn_box_l.addWidget(btn_apply_figi_l)

        btn_fetch_figi_l = QPushButton("🔍 Fetch OpenFIGI")
        btn_fetch_figi_l.setToolTip("Query OpenFIGI API to resolve ISIN")
        btn_fetch_figi_l.clicked.connect(self._fetch_openfigi_for_ledger_record)
        figi_btn_box_l.addWidget(btn_fetch_figi_l)
        figi_btn_box_l.addStretch()
        grid.addLayout(figi_btn_box_l, 3, 2, 1, 2)

        grid.addWidget(QLabel("Quantity:"), 3, 4)
        self._ledger_edit_quantity = QLineEdit()
        grid.addWidget(self._ledger_edit_quantity, 3, 5)

        grid.addWidget(QLabel("Unit Price:"), 3, 6)
        self._ledger_edit_unit_price = QLineEdit()
        grid.addWidget(self._ledger_edit_unit_price, 3, 7)

        grid.addWidget(QLabel("Currency:"), 4, 0)
        self._ledger_edit_currency = QLineEdit()
        grid.addWidget(self._ledger_edit_currency, 4, 1)

        grid.addWidget(QLabel("Fees:"), 4, 2)
        self._ledger_edit_fees = QLineEdit()
        grid.addWidget(self._ledger_edit_fees, 4, 3)

        grid.addWidget(QLabel("Total Amount:"), 4, 4)
        self._ledger_edit_total_amount = QLineEdit()
        grid.addWidget(self._ledger_edit_total_amount, 4, 5)

        grid.addWidget(QLabel("FX Rate (EUR):"), 4, 6)
        self._ledger_edit_fx_rate = QLineEdit()
        grid.addWidget(self._ledger_edit_fx_rate, 4, 7)

        grid.addWidget(QLabel("Local Total:"), 5, 0)
        self._ledger_edit_local_total = QLineEdit()
        grid.addWidget(self._ledger_edit_local_total, 5, 1)

        grid.addWidget(QLabel("Source File:"), 5, 2)
        self._ledger_edit_source_file = QLineEdit()
        self._ledger_edit_source_file.setReadOnly(True)
        grid.addWidget(self._ledger_edit_source_file, 5, 3, 1, 5)

        d_layout.addLayout(grid)

        btn_row = QHBoxLayout()
        self._btn_ledger_save = QPushButton("Save Ledger Edits")
        self._btn_ledger_save.clicked.connect(self._save_ledger_changes)
        btn_row.addWidget(self._btn_ledger_save)

        self._btn_ledger_delete = QPushButton("Delete Ledger Record")
        self._btn_ledger_delete.clicked.connect(self._delete_ledger_record)
        btn_row.addWidget(self._btn_ledger_delete)

        btn_row.addStretch()
        d_layout.addLayout(btn_row)

        splitter.addWidget(d_container)
        splitter.setSizes([380, 320])
        layout.addWidget(splitter)

    def load_ledger_records(self) -> None:
        """Fetch records from financial_records and apply current UI filters."""
        jur_filter = self._ledger_filter_account_country.currentText()
        jur = None if jur_filter == "All" else jur_filter
        search_query = self._ledger_filter_search.text().strip().lower()

        try:
            records = self._db.get_financial_records(account_country=jur)
        except Exception as exc:
            QMessageBox.critical(self, "Database Error", f"Failed to load ledger records:\n{exc}")
            return

        if search_query:
            filtered: list[FinancialRecord] = []
            for r in records:
                text_blob = f"{r.symbol or ''} {r.provider or ''} {r.isin or ''} {r.asset_name or ''} {r.openfigi_detected or ''}".lower()
                if search_query in text_blob:
                    filtered.append(r)
            records = filtered

        self._ledger_records = records
        self._populate_ledger_table()
        self.load_portfolio_snapshot()

    def _populate_ledger_table(self) -> None:
        self._table_ledger.setRowCount(0)
        for record in self._ledger_records:
            row = self._table_ledger.rowCount()
            self._table_ledger.insertRow(row)

            st_item = QTableWidgetItem("✅ Approved")
            st_item.setBackground(QColor(230, 255, 230))
            st_item.setForeground(QColor(0, 120, 0))
            self._table_ledger.setItem(row, 0, st_item)

            self._table_ledger.setItem(row, 1, QTableWidgetItem(str(record.id)))

            ts_str = record.event_timestamp.strftime("%Y-%m-%d %H:%M") if record.event_timestamp else "-"
            self._table_ledger.setItem(row, 2, QTableWidgetItem(ts_str))
            self._table_ledger.setItem(row, 3, QTableWidgetItem(record.provider or "-"))
            self._table_ledger.setItem(row, 4, QTableWidgetItem(record.account_country or "-"))
            self._table_ledger.setItem(row, 5, QTableWidgetItem(str(record.tax_year) if record.tax_year else "-"))
            self._table_ledger.setItem(row, 6, QTableWidgetItem(record.asset_type or "-"))

            sym_name = record.symbol or record.openfigi_detected or record.asset_name or record.isin or "-"
            self._table_ledger.setItem(row, 7, QTableWidgetItem(sym_name))
            self._table_ledger.setItem(row, 8, QTableWidgetItem(record.action or "-"))

            qty_str = f"{record.quantity:.4f}" if record.quantity is not None else "-"
            self._table_ledger.setItem(row, 9, QTableWidgetItem(qty_str))

            price_str = f"{record.unit_price:.2f}" if record.unit_price is not None else "-"
            self._table_ledger.setItem(row, 10, QTableWidgetItem(price_str))

            curr = record.currency or "EUR"
            tot_str = f"{record.total_amount:.2f} {curr}" if record.total_amount is not None else "-"
            self._table_ledger.setItem(row, 11, QTableWidgetItem(tot_str))

            loc_tot = record.local_total_amount
            loc_str = f"€{loc_tot:.2f}" if loc_tot is not None else "-"
            self._table_ledger.setItem(row, 12, QTableWidgetItem(loc_str))

            if record.fx_rate is None and (record.currency or "EUR").upper() == "EUR":
                record.fx_rate = Decimal("1.0")
            if record.local_total_amount is None and record.total_amount is not None and record.fx_rate is not None:
                record.local_total_amount = record.total_amount * record.fx_rate
            if record.fees is None:
                record.fees = Decimal("0.0")

            try:
                strict_obj = BaseStrictRecord.from_raw(record)
            except Exception:
                strict_obj = None

            strict_item = QTableWidgetItem("Yes" if strict_obj else "No")
            if not strict_obj:
                strict_item.setForeground(QColor(180, 0, 0))
            self._table_ledger.setItem(row, 13, strict_item)

            st_item.setData(Qt.ItemDataRole.UserRole, record)

    def _on_ledger_selection_changed(self) -> None:
        selected_rows = self._table_ledger.selectedItems()
        if not selected_rows:
            self._selected_ledger_record = None
            self._clear_ledger_form()
            return

        row = self._table_ledger.currentRow()
        item = self._table_ledger.item(row, 0)
        if item is not None:
            record: FinancialRecord = item.data(Qt.ItemDataRole.UserRole)
            self._selected_ledger_record = record
            self._populate_ledger_form(record)

    def _populate_ledger_form(self, record: FinancialRecord) -> None:
        self._ledger_edit_id.setText(str(record.id or ""))
        self._ledger_edit_provider.setText(record.provider or "")
        idx_jur = self._ledger_edit_account_country.findText(record.account_country or "italy")
        if idx_jur >= 0:
            self._ledger_edit_account_country.setCurrentIndex(idx_jur)

        self._ledger_edit_tax_year.setText(str(record.tax_year or ""))
        self._ledger_edit_timestamp.setText(record.event_timestamp.isoformat() if record.event_timestamp else "")

        idx_asset = self._ledger_edit_asset_type.findText(record.asset_type or "stock")
        if idx_asset >= 0:
            self._ledger_edit_asset_type.setCurrentIndex(idx_asset)

        idx_action = self._ledger_edit_action.findText(record.action or "buy")
        if idx_action >= 0:
            self._ledger_edit_action.setCurrentIndex(idx_action)

        self._ledger_edit_symbol.setText(record.symbol or "")
        self._ledger_edit_isin.setText(record.isin or "")
        self._ledger_edit_asset_name.setText(record.asset_name or "")
        self._ledger_edit_openfigi_detected.setText(record.openfigi_detected or "")
        self._ledger_edit_quantity.setText(str(record.quantity) if record.quantity is not None else "")
        self._ledger_edit_unit_price.setText(str(record.unit_price) if record.unit_price is not None else "")
        self._ledger_edit_currency.setText(record.currency or "EUR")
        self._ledger_edit_fees.setText(str(record.fees) if record.fees is not None else "")
        self._ledger_edit_total_amount.setText(str(record.total_amount) if record.total_amount is not None else "")
        self._ledger_edit_fx_rate.setText(str(record.fx_rate) if record.fx_rate is not None else "")

        loc_tot = record.local_total_amount
        self._ledger_edit_local_total.setText(str(loc_tot) if loc_tot is not None else "")
        self._ledger_edit_source_file.setText(record.source_file_name or "")

    def _clear_ledger_form(self) -> None:
        self._ledger_edit_id.clear()
        self._ledger_edit_provider.clear()
        self._ledger_edit_tax_year.clear()
        self._ledger_edit_timestamp.clear()
        self._ledger_edit_symbol.clear()
        self._ledger_edit_isin.clear()
        self._ledger_edit_asset_name.clear()
        self._ledger_edit_openfigi_detected.clear()
        self._ledger_edit_quantity.clear()
        self._ledger_edit_unit_price.clear()
        self._ledger_edit_currency.setText("EUR")
        self._ledger_edit_fees.clear()
        self._ledger_edit_total_amount.clear()
        self._ledger_edit_fx_rate.clear()
        self._ledger_edit_local_total.clear()
        self._ledger_edit_source_file.clear()

    def _apply_ledger_openfigi_symbol(self) -> None:
        figi_sym = self._ledger_edit_openfigi_detected.text().strip()
        if figi_sym:
            self._ledger_edit_symbol.setText(figi_sym)

    def _fetch_openfigi_for_ledger_record(self) -> None:
        if not self._selected_ledger_record:
            QMessageBox.warning(self, "No Selection", "Please select a ledger record first.")
            return

        isin = self._ledger_edit_isin.text().strip().upper()
        if not isin or len(isin) != 12:
            QMessageBox.warning(self, "Invalid ISIN", "Please enter a valid 12-character ISIN code.")
            return

        figi_res = self._figi_mapper.map_isin(isin)
        ticker, name = figi_res.ticker, figi_res.name
        if ticker:
            self._ledger_edit_openfigi_detected.setText(ticker)
            if not self._ledger_edit_symbol.text().strip():
                self._ledger_edit_symbol.setText(ticker)
            if name and not self._ledger_edit_asset_name.text().strip():
                self._ledger_edit_asset_name.setText(name)

            self._selected_ledger_record.openfigi_detected = ticker
            self._selected_ledger_record.symbol = self._ledger_edit_symbol.text().strip()
            self._selected_ledger_record.asset_name = self._ledger_edit_asset_name.text().strip()
            strict_rec = BaseStrictRecord.from_raw(self._selected_ledger_record)
            self._db.update_strict_financial_record(strict_rec)
            self.load_ledger_records()
            QMessageBox.information(
                self, "OpenFIGI Found", f"Resolved OpenFIGI Ticker: {ticker}\nCompany Name: {name or 'N/A'}"
            )
        else:
            QMessageBox.warning(self, "Not Found", f"OpenFIGI could not resolve ISIN: {isin}")

    def _backfill_openfigi_ledger(self) -> None:
        ledger_records = self._db.get_financial_records()
        count = 0
        for r in ledger_records:
            if r.isin and (not r.symbol or not r.openfigi_detected):
                figi_res = self._figi_mapper.map_isin(r.isin)
                ticker, name = figi_res.ticker, figi_res.name
                if ticker:
                    r.openfigi_detected = ticker
                    if not r.symbol:
                        r.symbol = ticker
                    if name and not r.asset_name:
                        r.asset_name = name
                    strict_r = BaseStrictRecord.from_raw(r)
                    self._db.update_strict_financial_record(strict_r)
                    count += 1

        self.load_ledger_records()
        QMessageBox.information(self, "Backfill Complete", f"Backfilled OpenFIGI tickers for {count} ledger record(s).")

    def _save_ledger_changes(self) -> None:
        if not self._selected_ledger_record:
            QMessageBox.warning(self, "No Selection", "Please select a ledger record to edit.")
            return

        rec = self._selected_ledger_record
        rec.provider = self._ledger_edit_provider.text().strip() or None
        rec.account_country = self._ledger_edit_account_country.currentText()
        rec.tax_year = self._parse_int(self._ledger_edit_tax_year.text())
        rec.event_timestamp = self._parse_datetime(self._ledger_edit_timestamp.text()) or rec.event_timestamp
        rec.asset_type = self._ledger_edit_asset_type.currentText()
        rec.action = self._ledger_edit_action.currentText()
        rec.symbol = self._ledger_edit_symbol.text().strip() or None
        rec.isin = self._ledger_edit_isin.text().strip() or None
        rec.asset_name = self._ledger_edit_asset_name.text().strip() or None
        rec.openfigi_detected = self._ledger_edit_openfigi_detected.text().strip() or None
        rec.quantity = self._parse_decimal(self._ledger_edit_quantity.text())
        rec.unit_price = self._parse_decimal(self._ledger_edit_unit_price.text())
        rec.currency = self._ledger_edit_currency.text().strip().upper() or "EUR"
        rec.fees = self._parse_decimal(self._ledger_edit_fees.text())
        rec.total_amount = self._parse_decimal(self._ledger_edit_total_amount.text())
        rec.fx_rate = self._parse_decimal(self._ledger_edit_fx_rate.text())
        rec.local_total_amount = self._parse_decimal(self._ledger_edit_local_total.text())

        if rec.local_total_amount is None and rec.total_amount is not None:
            if rec.currency == "EUR":
                rec.local_total_amount = rec.total_amount
            elif rec.fx_rate is not None:
                rec.local_total_amount = rec.total_amount * rec.fx_rate

        try:
            strict_rec = BaseStrictRecord.from_raw(rec)
            updated = self._db.update_strict_financial_record(strict_rec)
            self._selected_ledger_record = updated.to_raw()
            QMessageBox.information(self, "Ledger Updated", f"Successfully updated record #{updated.id}.")
            self.load_ledger_records()
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save ledger record:\n{exc}")

    def _delete_ledger_record(self) -> None:
        if not self._selected_ledger_record or self._selected_ledger_record.id is None:
            return
        rec = self._selected_ledger_record
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to permanently delete approved record #{rec.id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if rec.id is not None:
                self._db.delete_financial_record_by_id(rec.id)
            self._selected_ledger_record = None
            self._clear_ledger_form()
            self.load_ledger_records()

    # ------------------------------------------------------------------
    # Tab 3: Portfolio Snapshot
    # ------------------------------------------------------------------

    def _init_portfolio_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addLayout(self._build_portfolio_filter_bar())
        layout.addLayout(self._build_portfolio_summary_cards())

        splitter = QSplitter(Qt.Orientation.Vertical, tab)
        splitter.setHandleWidth(4)

        pos_container = QWidget()
        pos_layout = QVBoxLayout(pos_container)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setSpacing(4)
        pos_layout.addWidget(_section_label("Portfolio Holdings (Snapshot at Today Date)"))

        self._table_portfolio = QTableWidget(0, 14)
        headers = [
            "Status",
            "Symbol",
            "Asset Name",
            "ISIN",
            "Type",
            "Provider",
            "Account Country",
            "Holdings Qty",
            "Avg Buy Price",
            "Cost Basis (€)",
            "Realized P&L (€)",
            "Dividends (€)",
            "Last Activity",
            "Tx Count",
        ]
        self._table_portfolio.setHorizontalHeaderLabels(headers)
        hdr_item = self._table_portfolio.horizontalHeaderItem(9)
        if hdr_item is not None:
            hdr_item.setToolTip(
                "Cost basis computed as current holdings quantity multiplied by average buy price (WAC / PMP)."
            )

        self._table_portfolio.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_portfolio.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_portfolio.setAlternatingRowColors(True)
        self._table_portfolio.itemSelectionChanged.connect(self._on_portfolio_table_selection_changed)

        h_port = self._table_portfolio.horizontalHeader()
        h_port.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_port.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_port.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        pos_layout.addWidget(self._table_portfolio)
        splitter.addWidget(pos_container)

        tx_container = QWidget()
        tx_layout = QVBoxLayout(tx_container)
        tx_layout.setContentsMargins(0, 4, 0, 0)
        tx_layout.setSpacing(4)

        self._lbl_holding_txs_header = _section_label("Underlying Transactions (Select a holding position above)")
        tx_layout.addWidget(self._lbl_holding_txs_header)

        self._table_holding_txs = QTableWidget(0, 9)
        tx_headers = [
            "ID",
            "Date",
            "Provider",
            "Action",
            "Qty",
            "Unit Price",
            "Total Amount",
            "Local Total (€)",
            "Status",
        ]
        self._table_holding_txs.setHorizontalHeaderLabels(tx_headers)
        self._table_holding_txs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_holding_txs.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_holding_txs.setAlternatingRowColors(True)

        tx_layout.addWidget(self._table_holding_txs)
        splitter.addWidget(tx_container)

        splitter.setSizes([400, 240])
        layout.addWidget(splitter)

    def _build_portfolio_filter_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("Account Country:"))
        self._port_filter_account_country = QComboBox()
        self._port_filter_account_country.addItems(["All", "italy", "ireland"])
        self._port_filter_account_country.currentTextChanged.connect(self.load_portfolio_snapshot)
        bar.addWidget(self._port_filter_account_country)

        bar.addWidget(QLabel("Provider:"))
        self._port_filter_provider = QComboBox()
        self._port_filter_provider.addItems(["All", "directa", "interactive_brokers", "f24", "revenue", "manual"])
        self._port_filter_provider.currentTextChanged.connect(self.load_portfolio_snapshot)
        bar.addWidget(self._port_filter_provider)

        bar.addWidget(QLabel("Asset Type:"))
        self._port_filter_asset_type = QComboBox()
        self._port_filter_asset_type.addItems(["All", "stock", "etf", "cash", "tax_payment", "salary", "pension"])
        self._port_filter_asset_type.currentTextChanged.connect(self.load_portfolio_snapshot)
        bar.addWidget(self._port_filter_asset_type)

        bar.addWidget(QLabel("Status:"))
        self._port_filter_status = QComboBox()
        self._port_filter_status.addItems(
            ["Active Positions Only", "All Positions (incl. closed)", "Closed Positions Only"]
        )
        self._port_filter_status.currentTextChanged.connect(self.load_portfolio_snapshot)
        bar.addWidget(self._port_filter_status)

        bar.addWidget(QLabel("Search:"))
        self._port_filter_search = QLineEdit()
        self._port_filter_search.setPlaceholderText("Symbol, ISIN, asset name...")
        self._port_filter_search.textChanged.connect(self.load_portfolio_snapshot)
        bar.addWidget(self._port_filter_search)

        btn_port_refresh = QPushButton("Refresh")
        btn_port_refresh.clicked.connect(self.load_portfolio_snapshot)
        bar.addWidget(btn_port_refresh)

        self._lbl_port_latency = QLabel("⚡ Computed on-the-fly in 0.0 ms")
        self._lbl_port_latency.setStyleSheet("color: #666; font-size: 11px; font-weight: bold;")
        bar.addWidget(self._lbl_port_latency)

        bar.addStretch()
        return bar

    def _build_portfolio_summary_cards(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        def _create_card(title: str) -> tuple[QFrame, QLabel]:
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setStyleSheet(
                "QFrame { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 4px 8px; }"
            )
            l = QVBoxLayout(card)
            l.setContentsMargins(6, 4, 6, 4)
            lbl_title = QLabel(title)
            lbl_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #555;")
            lbl_val = QLabel("€0.00")
            lbl_val.setStyleSheet("font-size: 15px; font-weight: bold; color: #111;")
            l.addWidget(lbl_title)
            l.addWidget(lbl_val)
            return card, lbl_val

        card1, self._lbl_card_active = _create_card("ACTIVE POSITIONS")
        card2, self._lbl_card_invested = _create_card("TOTAL COST BASIS (€)")
        card3, self._lbl_card_realized = _create_card("REALIZED P&L (€)")
        card4, self._lbl_card_dividends = _create_card("TOTAL DIVIDENDS (€)")

        row.addWidget(card1)
        row.addWidget(card2)
        row.addWidget(card3)
        row.addWidget(card4)
        return row

    def load_portfolio_snapshot(self) -> None:
        """Reconstruct portfolio snapshot from approved database records on-the-fly."""
        st_text = self._port_filter_status.currentText()
        pos_st: str = "active"
        if "All" in st_text:
            pos_st = "all"
        elif "Closed" in st_text:
            pos_st = "closed"

        filters = PortfolioFilter(
            account_country=self._port_filter_account_country.currentText(),
            provider=self._port_filter_provider.currentText(),
            asset_type=self._port_filter_asset_type.currentText(),
            position_status=pos_st,
            search_query=self._port_filter_search.text(),
        )

        try:
            snapshot = PortfolioEngine.get_snapshot(self._db, filters)
            self._portfolio_snapshot = snapshot
            self._populate_portfolio_table(snapshot)
        except Exception as exc:
            QMessageBox.critical(self, "Portfolio Error", f"Failed to compute portfolio snapshot:\n{exc}")

    def _populate_portfolio_table(self, snapshot: PortfolioSnapshot) -> None:
        self._lbl_card_active.setText(str(snapshot.total_active_positions))
        self._lbl_card_invested.setText(f"€{snapshot.total_cost_basis:,.2f}")

        pnl = snapshot.total_realized_pnl
        pnl_str = f"€{pnl:+,.2f}" if pnl != Decimal("0") else f"€{pnl:,.2f}"
        self._lbl_card_realized.setText(pnl_str)
        if pnl > Decimal("0"):
            self._lbl_card_realized.setStyleSheet("font-size: 15px; font-weight: bold; color: #008000;")
        elif pnl < Decimal("0"):
            self._lbl_card_realized.setStyleSheet("font-size: 15px; font-weight: bold; color: #cc0000;")
        else:
            self._lbl_card_realized.setStyleSheet("font-size: 15px; font-weight: bold; color: #111;")

        self._lbl_card_dividends.setText(f"€{snapshot.total_dividends:,.2f}")
        self._lbl_port_latency.setText(f"⚡ Computed on-the-fly in {snapshot.elapsed_ms:.2f} ms")

        self._table_portfolio.setRowCount(0)
        for pos in snapshot.positions:
            row = self._table_portfolio.rowCount()
            self._table_portfolio.insertRow(row)

            is_active = pos.current_quantity > Decimal("0")
            st_item = QTableWidgetItem("🟢 Active" if is_active else "🔴 Closed")
            if is_active:
                st_item.setBackground(QColor(230, 255, 230))
            else:
                st_item.setBackground(QColor(240, 240, 240))
                st_item.setForeground(QColor(120, 120, 120))
            self._table_portfolio.setItem(row, 0, st_item)

            self._table_portfolio.setItem(row, 1, QTableWidgetItem(pos.symbol or "-"))
            self._table_portfolio.setItem(row, 2, QTableWidgetItem(pos.asset_name or "-"))
            self._table_portfolio.setItem(row, 3, QTableWidgetItem(pos.isin or "-"))
            self._table_portfolio.setItem(row, 4, QTableWidgetItem(pos.asset_type or "-"))
            self._table_portfolio.setItem(row, 5, QTableWidgetItem(pos.provider or "-"))
            self._table_portfolio.setItem(row, 6, QTableWidgetItem(pos.account_country or "-"))
            self._table_portfolio.setItem(row, 7, QTableWidgetItem(f"{pos.current_quantity:.4f}"))

            price_curr = pos.currency or "EUR"
            self._table_portfolio.setItem(row, 8, QTableWidgetItem(f"{pos.average_buy_price:.2f} {price_curr}"))
            cost_item = QTableWidgetItem(f"€{pos.cost_basis:,.2f}")
            cost_item.setToolTip(
                f"Cost basis: {pos.current_quantity:.4f} shares × €{pos.average_buy_price:.2f} (WAC / PMP)"
            )
            self._table_portfolio.setItem(row, 9, cost_item)

            pnl_item = QTableWidgetItem(
                f"€{pos.realized_pnl:+,.2f}" if pos.realized_pnl != Decimal("0") else f"€{pos.realized_pnl:,.2f}"
            )
            if pos.realized_pnl > Decimal("0"):
                pnl_item.setForeground(QColor(0, 120, 0))
            elif pos.realized_pnl < Decimal("0"):
                pnl_item.setForeground(QColor(180, 0, 0))
            self._table_portfolio.setItem(row, 10, pnl_item)

            div_item = QTableWidgetItem(f"€{pos.total_dividends:,.2f}")
            if pos.total_dividends > Decimal("0"):
                div_item.setForeground(QColor(0, 120, 0))
            self._table_portfolio.setItem(row, 11, div_item)

            last_date_str = pos.last_activity_date.strftime("%Y-%m-%d") if pos.last_activity_date else "-"
            self._table_portfolio.setItem(row, 12, QTableWidgetItem(last_date_str))
            self._table_portfolio.setItem(row, 13, QTableWidgetItem(str(pos.transaction_count)))

            st_item.setData(Qt.ItemDataRole.UserRole, pos)

    def _on_portfolio_table_selection_changed(self) -> None:
        row = self._table_portfolio.currentRow()
        if row < 0:
            self._table_holding_txs.setRowCount(0)
            self._lbl_holding_txs_header.setText("Underlying Transactions (Select a holding position above)")
            return

        st_item = self._table_portfolio.item(row, 0)
        if not st_item:
            return

        pos: PortfolioPosition = st_item.data(Qt.ItemDataRole.UserRole)
        if not pos:
            return

        self._lbl_holding_txs_header.setText(
            f"Underlying Transactions for {pos.symbol or pos.asset_name or 'Position'} ({pos.provider} / {pos.account_country})"
        )

        all_records = self._db.get_financial_records()
        matching_records: list[FinancialRecord] = []

        for rec in all_records:
            sym_match = (rec.symbol or "").lower() == (pos.symbol or "").lower() if pos.symbol else True
            isin_match = (rec.isin or "").lower() == (pos.isin or "").lower() if pos.isin else True
            prov_match = (rec.provider or "").lower() == (pos.provider or "").lower() if pos.provider else True
            jur_match = (
                (rec.account_country or "").lower() == (pos.account_country or "").lower() if pos.account_country else True
            )

            if (
                (pos.symbol and sym_match and prov_match and jur_match)
                or (pos.isin and isin_match and prov_match and jur_match)
                or (rec.asset_name and pos.asset_name and rec.asset_name.lower() == pos.asset_name.lower())
            ):
                matching_records.append(rec)

        self._table_holding_txs.setRowCount(0)
        for rec in matching_records:
            r_idx = self._table_holding_txs.rowCount()
            self._table_holding_txs.insertRow(r_idx)

            self._table_holding_txs.setItem(r_idx, 0, QTableWidgetItem(str(rec.id)))
            ts_str = rec.event_timestamp.strftime("%Y-%m-%d %H:%M") if rec.event_timestamp else "-"
            self._table_holding_txs.setItem(r_idx, 1, QTableWidgetItem(ts_str))
            self._table_holding_txs.setItem(r_idx, 2, QTableWidgetItem(rec.provider or "-"))
            self._table_holding_txs.setItem(r_idx, 3, QTableWidgetItem(rec.action or "-"))
            self._table_holding_txs.setItem(
                r_idx, 4, QTableWidgetItem(f"{rec.quantity:.4f}" if rec.quantity is not None else "-")
            )
            self._table_holding_txs.setItem(
                r_idx, 5, QTableWidgetItem(f"{rec.unit_price:.2f}" if rec.unit_price is not None else "-")
            )
            self._table_holding_txs.setItem(
                r_idx,
                6,
                QTableWidgetItem(f"{rec.total_amount:.2f} {rec.currency}" if rec.total_amount is not None else "-"),
            )
            self._table_holding_txs.setItem(
                r_idx,
                7,
                QTableWidgetItem(f"€{rec.local_total_amount:.2f}" if rec.local_total_amount is not None else "-"),
            )
            self._table_holding_txs.setItem(r_idx, 8, QTableWidgetItem(rec.verification_status or "-"))

    # ------------------------------------------------------------------
    # Parsing Helpers
    # ------------------------------------------------------------------

    def _parse_decimal(self, text: str) -> Decimal | None:
        if not text:
            return None
        s = text.strip().replace("€", "").replace(" ", "").replace("$", "")
        if not s:
            return None
        if "." in s and "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return Decimal(s)
        except Exception:
            return None

    def _parse_int(self, text: str) -> int | None:
        s = text.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def _parse_datetime(self, text: str) -> datetime | None:
        s = text.strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Tab 4: ETF Mergers & Corporate Actions
    # ------------------------------------------------------------------

    def _init_mergers_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Splitter: Table on Top, Registration Form on Bottom
        splitter = QSplitter(Qt.Orientation.Vertical, tab)
        splitter.setHandleWidth(4)

        # Section 1: Table
        table_container = QWidget()
        t_layout = QVBoxLayout(table_container)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(4)
        t_layout.addWidget(_section_label("Registered ETF Corporate Mergers & Reorganizations"))

        self._table_mergers = QTableWidget(0, 9)
        headers = [
            "ID",
            "Old ISIN",
            "New ISIN",
            "Old Symbol",
            "New Symbol",
            "Effective Date",
            "Exchange Ratio",
            "Old NAV (€)",
            "New NAV (€)",
        ]
        self._table_mergers.setHorizontalHeaderLabels(headers)
        self._table_mergers.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_mergers.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_mergers.setAlternatingRowColors(True)

        h = self._table_mergers.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        t_layout.addWidget(self._table_mergers)
        splitter.addWidget(table_container)

        # Section 2: Registration Form
        form_container = QWidget()
        f_layout = QVBoxLayout(form_container)
        f_layout.setContentsMargins(0, 4, 0, 0)
        f_layout.setSpacing(6)

        f_layout.addWidget(_section_label("➕ Register New ETF Merger / Absorption"))

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Old ISIN (Absorbed ETF):"), 0, 0)
        self._merger_old_isin = QLineEdit()
        self._merger_old_isin.setPlaceholderText("e.g. LU1781541179")
        grid.addWidget(self._merger_old_isin, 0, 1)

        grid.addWidget(QLabel("New ISIN (Surviving ETF):"), 0, 2)
        self._merger_new_isin = QLineEdit()
        self._merger_new_isin.setPlaceholderText("e.g. IE000BI8OT95")
        grid.addWidget(self._merger_new_isin, 0, 3)

        grid.addWidget(QLabel("Valuation Date:"), 0, 4)
        self._merger_effective_date = QLineEdit()
        self._merger_effective_date.setPlaceholderText("e.g. 2025-02-20")
        grid.addWidget(self._merger_effective_date, 0, 5)

        # Fetch NAVs button
        btn_fetch = QPushButton("🔍 Fetch NAVs & Auto-Calculate")
        btn_fetch.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 4px 10px;")
        btn_fetch.clicked.connect(self._fetch_merger_navs)
        grid.addWidget(btn_fetch, 0, 6)

        grid.addWidget(QLabel("Old Ticker:"), 1, 0)
        self._merger_old_symbol = QLineEdit()
        self._merger_old_symbol.setPlaceholderText("e.g. LCWD")
        grid.addWidget(self._merger_old_symbol, 1, 1)

        grid.addWidget(QLabel("New Ticker:"), 1, 2)
        self._merger_new_symbol = QLineEdit()
        self._merger_new_symbol.setPlaceholderText("e.g. MWRD")
        grid.addWidget(self._merger_new_symbol, 1, 3)

        grid.addWidget(QLabel("Old NAV (€):"), 1, 4)
        self._merger_old_nav = QLineEdit()
        self._merger_old_nav.setPlaceholderText("e.g. 19.81")
        grid.addWidget(self._merger_old_nav, 1, 5)

        grid.addWidget(QLabel("New NAV (€):"), 1, 6)
        self._merger_new_nav = QLineEdit()
        self._merger_new_nav.setPlaceholderText("e.g. 136.52")
        grid.addWidget(self._merger_new_nav, 1, 7)

        # Compute button
        btn_compute = QPushButton("🧮 Compute Ratio & Breakdown")
        btn_compute.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 4px 10px;")
        btn_compute.clicked.connect(self._compute_merger_breakdown)
        grid.addWidget(btn_compute, 2, 0)

        grid.addWidget(QLabel("Exchange Ratio (Auto):"), 2, 1)
        self._merger_ratio = QLineEdit()
        self._merger_ratio.setPlaceholderText("Auto-computed from NAVs or ratio input")
        grid.addWidget(self._merger_ratio, 2, 2)

        grid.addWidget(QLabel("Calculated Old Qty:"), 2, 3)
        self._merger_diag_old_qty = QLineEdit()
        self._merger_diag_old_qty.setReadOnly(True)
        grid.addWidget(self._merger_diag_old_qty, 2, 4)

        grid.addWidget(QLabel("Whole Shares Received:"), 2, 5)
        self._merger_diag_whole_shares = QLineEdit()
        self._merger_diag_whole_shares.setPlaceholderText("e.g. 275")
        grid.addWidget(self._merger_diag_whole_shares, 2, 6)

        grid.addWidget(QLabel("Fractional Shares:"), 3, 0)
        self._merger_diag_fractional_shares = QLineEdit()
        self._merger_diag_fractional_shares.setPlaceholderText("e.g. 0.1346")
        grid.addWidget(self._merger_diag_fractional_shares, 3, 1)

        grid.addWidget(QLabel("Expected Cash Payout (€):"), 3, 2)
        self._merger_diag_cash_payout = QLineEdit()
        self._merger_diag_cash_payout.setPlaceholderText("e.g. 18.38")
        grid.addWidget(self._merger_diag_cash_payout, 3, 3)

        f_layout.addLayout(grid)

        # Buttons
        btn_box = QHBoxLayout()
        btn_add = QPushButton("💾 Save ETF Merger")
        btn_add.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 5px 12px;")
        btn_add.clicked.connect(self._add_asset_merger)
        btn_box.addWidget(btn_add)

        btn_del = QPushButton("🗑️ Delete Selected Merger")
        btn_del.clicked.connect(self._delete_asset_merger)
        btn_box.addWidget(btn_del)

        btn_ref = QPushButton("Refresh List")
        btn_ref.clicked.connect(self.load_asset_mergers)
        btn_box.addWidget(btn_ref)

        btn_box.addStretch()
        f_layout.addLayout(btn_box)

        splitter.addWidget(form_container)
        splitter.setSizes([380, 270])
        layout.addWidget(splitter)

    def _compute_merger_breakdown(self) -> None:
        """Compute exchange ratio, whole shares, fractional shares, and expected cash payout from user inputs."""
        old_isin = self._merger_old_isin.text().strip().upper()
        new_isin = self._merger_new_isin.text().strip().upper()
        date_str = self._merger_effective_date.text().strip()
        old_nav = self._parse_decimal(self._merger_old_nav.text())
        new_nav = self._parse_decimal(self._merger_new_nav.text())
        direct_ratio = self._parse_decimal(self._merger_ratio.text())
        user_whole_shares = self._parse_decimal(self._merger_diag_whole_shares.text())
        user_cash_payout = self._parse_decimal(self._merger_diag_cash_payout.text())

        if not old_isin or not new_isin:
            QMessageBox.warning(self, "Missing Input", "Please enter Old ISIN and New ISIN.")
            return

        if date_str:
            parsed_dt = self._parse_datetime(date_str)
            if parsed_dt is None:
                try:
                    eff_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Date", "Please enter date in format YYYY-MM-DD (e.g. 2025-02-20).")
                    return
            else:
                eff_date = parsed_dt
        else:
            eff_date = datetime.now(timezone.utc)

        # Query database holdings of old_isin prior to eff_date
        records = self._db.get_financial_records()
        old_qty = Decimal("0")
        for r in records:
            if r.isin and r.isin.upper() == old_isin and r.event_timestamp and r.event_timestamp <= eff_date:
                if r.action and r.action.lower() in ("buy", "acquisto"):
                    old_qty += r.quantity or Decimal("0")
                elif r.action and r.action.lower() in ("sell", "vendita"):
                    old_qty -= r.quantity or Decimal("0")

        self._merger_diag_old_qty.setText(f"{old_qty:.4f}")

        # Determine ratio
        exchange_ratio = direct_ratio
        if exchange_ratio is None and old_nav is not None and new_nav is not None and new_nav > 0:
            exchange_ratio = (old_nav / new_nav).quantize(Decimal("0.000001"))

        # If user entered whole shares (e.g. 275) and cash payout (e.g. 18.38) with new_nav (136.52)
        if exchange_ratio is None and user_whole_shares is not None and old_qty > 0:
            fractional_portion = Decimal("0")
            if user_cash_payout is not None and new_nav is not None and new_nav > 0:
                fractional_portion = user_cash_payout / new_nav
            total_target_new_qty = user_whole_shares + fractional_portion
            exchange_ratio = (total_target_new_qty / old_qty).quantize(Decimal("0.000001"))

        if exchange_ratio is None:
            QMessageBox.warning(
                self, "Missing Data", "Please enter (Old NAV and New NAV), OR Exchange Ratio, OR Whole Shares Received."
            )
            return

        self._merger_ratio.setText(f"{exchange_ratio:.6f}")

        # Calculate share breakdown
        total_new_qty = old_qty * exchange_ratio
        whole_shares = user_whole_shares if user_whole_shares is not None else Decimal(int(total_new_qty))
        fractional_shares = total_new_qty - whole_shares
        fractional_shares = max(fractional_shares, Decimal("0"))

        expected_cash_payout = (
            user_cash_payout
            if user_cash_payout is not None
            else ((fractional_shares * new_nav) if new_nav else Decimal("0"))
        )

        self._merger_diag_whole_shares.setText(f"{whole_shares}")
        self._merger_diag_fractional_shares.setText(f"{fractional_shares:.6f}")
        self._merger_diag_cash_payout.setText(f"€{expected_cash_payout:.2f}")

        QMessageBox.information(
            self,
            "Merger Calculation Complete",
            f"Calculated Details:\n"
            f"• Old Qty: {old_qty:.4f}\n"
            f"• Exchange Ratio: {exchange_ratio:.6f}\n"
            f"• Whole Shares Received: {whole_shares}\n"
            f"• Fractional Shares: {fractional_shares:.6f}\n"
            f"• Expected Cash Payout: €{expected_cash_payout:.2f}",
        )

    def _fetch_merger_navs(self) -> None:
        """Fetch historical NAVs via Euronext / yfinance and auto-populate form fields."""
        old_isin = self._merger_old_isin.text().strip().upper()
        new_isin = self._merger_new_isin.text().strip().upper()
        date_str = self._merger_effective_date.text().strip()

        if not old_isin or not new_isin or not date_str:
            QMessageBox.warning(
                self, "Missing Input", "Please enter Old ISIN, New ISIN, and Valuation Date (YYYY-MM-DD)."
            )
            return

        eff_date = self._parse_datetime(date_str)
        if eff_date is None:
            try:
                eff_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Date", "Please enter valuation date in format YYYY-MM-DD (e.g. 2025-02-20)."
                )
                return

        records = self._db.get_financial_records()
        old_qty = Decimal("0")
        for r in records:
            if r.isin and r.isin.upper() == old_isin and r.event_timestamp and r.event_timestamp <= eff_date:
                if r.action and r.action.lower() in ("buy", "acquisto"):
                    old_qty += r.quantity or Decimal("0")
                elif r.action and r.action.lower() in ("sell", "vendita"):
                    old_qty -= r.quantity or Decimal("0")

        old_sym_override = self._merger_old_symbol.text().strip().upper() or None
        new_sym_override = self._merger_new_symbol.text().strip().upper() or None
        old_nav_override = self._parse_decimal(self._merger_old_nav.text())
        new_nav_override = self._parse_decimal(self._merger_new_nav.text())

        from backend.ingestion.nav_resolver import NAVResolver

        resolver = NAVResolver()
        res = resolver.resolve_merger_details(
            old_isin=old_isin,
            new_isin=new_isin,
            valuation_date=eff_date,
            old_quantity=old_qty if old_qty > 0 else None,
            old_symbol_override=old_sym_override,
            new_symbol_override=new_sym_override,
            old_nav_override=old_nav_override,
            new_nav_override=new_nav_override,
        )

        if res.old_symbol:
            self._merger_old_symbol.setText(res.old_symbol)
        if res.new_symbol:
            self._merger_new_symbol.setText(res.new_symbol)

        if res.nav_old is not None:
            self._merger_old_nav.setText(f"{res.nav_old:.4f}")
        if res.nav_new is not None:
            self._merger_new_nav.setText(f"{res.nav_new:.4f}")

        if res.exchange_ratio is not None:
            self._merger_ratio.setText(f"{res.exchange_ratio:.6f}")

        self._merger_diag_old_qty.setText(f"{res.old_quantity:.4f}")
        self._merger_diag_whole_shares.setText(f"{res.whole_shares}")
        self._merger_diag_fractional_shares.setText(f"{res.fractional_shares:.6f}")
        self._merger_diag_cash_payout.setText(f"€{res.expected_cash_payout:.2f}")

        old_nav_str = f"€{res.nav_old:.4f}" if res.nav_old is not None else "Not Found (Enter manually)"
        new_nav_str = f"€{res.nav_new:.4f}" if res.nav_new is not None else "Not Found (Enter manually)"

        QMessageBox.information(
            self,
            "NAV Calculation Complete",
            f"Calculated Details:\n"
            f"• Old NAV ({res.old_symbol}): {old_nav_str}\n"
            f"• New NAV ({res.new_symbol}): {new_nav_str}\n"
            f"• Ratio: {res.exchange_ratio or 'N/A'}\n"
            f"• Expected Whole Shares: {res.whole_shares}\n"
            f"• Expected Cash Payout: €{res.expected_cash_payout:.2f}",
        )

    def load_asset_mergers(self) -> None:
        """Fetch and display registered ETF mergers from database."""
        try:
            mergers = self._db.get_asset_mergers()
        except Exception as exc:
            QMessageBox.critical(self, "Database Error", f"Failed to load ETF mergers:\n{exc}")
            return

        self._table_mergers.setRowCount(len(mergers))
        for r_idx, m in enumerate(mergers):
            self._table_mergers.setItem(r_idx, 0, QTableWidgetItem(str(m.id or "")))
            self._table_mergers.setItem(r_idx, 1, QTableWidgetItem(m.old_isin))
            self._table_mergers.setItem(r_idx, 2, QTableWidgetItem(m.new_isin))
            self._table_mergers.setItem(r_idx, 3, QTableWidgetItem(m.old_symbol or "-"))
            self._table_mergers.setItem(r_idx, 4, QTableWidgetItem(m.new_symbol or "-"))
            self._table_mergers.setItem(r_idx, 5, QTableWidgetItem(m.effective_date.strftime("%Y-%m-%d")))
            self._table_mergers.setItem(r_idx, 6, QTableWidgetItem(f"{m.exchange_ratio:.6f}"))
            self._table_mergers.setItem(
                r_idx, 7, QTableWidgetItem(f"€{m.old_nav:.2f}" if m.old_nav is not None else "-")
            )
            self._table_mergers.setItem(
                r_idx, 8, QTableWidgetItem(f"€{m.new_nav:.2f}" if m.new_nav is not None else "-")
            )

            item_0 = self._table_mergers.item(r_idx, 0)
            if item_0:
                item_0.setData(Qt.ItemDataRole.UserRole, m)

    def _add_asset_merger(self) -> None:
        old_isin = self._merger_old_isin.text().strip().upper()
        new_isin = self._merger_new_isin.text().strip().upper()
        old_sym = self._merger_old_symbol.text().strip().upper() or None
        new_sym = self._merger_new_symbol.text().strip().upper() or None
        date_str = self._merger_effective_date.text().strip()
        ratio_dec = self._parse_decimal(self._merger_ratio.text()) or Decimal("1.0")
        old_nav = self._parse_decimal(self._merger_old_nav.text())
        new_nav = self._parse_decimal(self._merger_new_nav.text())

        if not old_isin or not new_isin:
            QMessageBox.warning(self, "Missing Fields", "Please enter both Old ISIN and New ISIN.")
            return

        eff_date = self._parse_datetime(date_str)
        if eff_date is None:
            try:
                eff_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                QMessageBox.warning(self, "Invalid Date", "Please enter valuation date in format YYYY-MM-DD.")
                return

        merger = AssetMerger(
            old_isin=old_isin,
            new_isin=new_isin,
            old_symbol=old_sym,
            new_symbol=new_sym,
            effective_date=eff_date,
            exchange_ratio=ratio_dec,
            old_nav=old_nav,
            new_nav=new_nav,
        )

        try:
            inserted = self._db.insert_asset_merger(merger)
            QMessageBox.information(
                self,
                "Merger Registered",
                f"✅ Registered merger #{inserted.id}: {old_isin} → {new_isin} (Ratio: {ratio_dec}).",
            )
            self._merger_old_isin.clear()
            self._merger_new_isin.clear()
            self._merger_old_symbol.clear()
            self._merger_new_symbol.clear()
            self._merger_effective_date.clear()
            self._merger_old_nav.clear()
            self._merger_new_nav.clear()
            self._merger_ratio.setText("1.0")
            self._merger_diag_old_qty.clear()
            self._merger_diag_whole_shares.clear()
            self._merger_diag_fractional_shares.clear()
            self._merger_diag_cash_payout.clear()
            self.load_asset_mergers()
            self.load_portfolio_snapshot()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to register asset merger:\n{exc}")

    def _delete_asset_merger(self) -> None:
        selected = self._table_mergers.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an ETF merger row from the table to delete.")
            return

        item = self._table_mergers.item(selected[0].row(), 0)
        if not item:
            return
        merger: AssetMerger | None = item.data(Qt.ItemDataRole.UserRole)
        if not merger or merger.id is None:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete corporate merger #{merger.id} ({merger.old_isin} → {merger.new_isin})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.delete_asset_merger(merger.id)
            self.load_asset_mergers()
            self.load_portfolio_snapshot()
