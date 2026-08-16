"""Side-by-side 3-Voter Diff & Merge Dialog for PySide6 UI."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from backend.db_manager import DatabaseManager
from backend.db_models import FinancialRecord, StagedFinancialRecord
from backend.domain_models import BaseStrictRecord

RecordType = FinancialRecord | StagedFinancialRecord
FieldGetter = Callable[[RecordType], str]
FieldSpec = tuple[str, str, FieldGetter]


class VoterDiffMergeDialog(QDialog):
    """Interactive 3-Voter Diff and Merge Dialog.

    Allows user to side-by-side inspect Voter 1, Voter 2, Voter 3 candidate outputs,
    see color-coded field mismatches, edit fields, and save approved records to DB.
    """

    def __init__(
        self, record: FinancialRecord | StagedFinancialRecord, db: DatabaseManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.record: FinancialRecord | StagedFinancialRecord = record
        self.db: DatabaseManager = db

        self.setWindowTitle(
            f"🏛️ Voter Consensus Diff & Merge - Record #{record.id} ({record.source_file_name or 'Document'})"
        )
        self.resize(1100, 650)

        # Parse consensus_log JSON
        self.consensus_log_data: dict[str, object] = {}
        if record.consensus_log:
            try:
                parsed = json.loads(record.consensus_log)
                if isinstance(parsed, dict):
                    self.consensus_log_data = parsed
            except Exception:
                self.consensus_log_data = {}

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Header Info
        header_text = (
            f"<b>Source File:</b> {self.record.source_file_name or 'N/A'} | "
            f"<b>Provider:</b> {self.record.provider or 'N/A'} | "
            f"<b>Status:</b> <font color='#e67e22'>{self.record.verification_status.upper()}</font>"
        )
        if self.consensus_log_data.get("error"):
            header_text += (
                f"<br/><font color='#c0392b'><b>Consensus Alert:</b> {self.consensus_log_data.get('error')}</font>"
            )

        info_label = QLabel(header_text)
        info_label.setStyleSheet("font-size: 13px; margin-bottom: 8px;")
        layout.addWidget(info_label)

        # Main Table: 3 Voters side-by-side comparison
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            [
                "Field",
                "Voter 1 Extraction",
                "Voter 2 Extraction",
                "Voter 3 Extraction",
                "Final Merged Value",
                "Apply V1",
                "Apply V2",
                "Apply V3",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        self._populate_diff_table()

        # Action Buttons
        btn_layout = QHBoxLayout()

        self.btn_apply_v1_all = QPushButton("Use All Voter 1 Values")
        self.btn_apply_v1_all.clicked.connect(lambda: self._apply_voter_all(1))
        btn_layout.addWidget(self.btn_apply_v1_all)

        self.btn_apply_v2_all = QPushButton("Use All Voter 2 Values")
        self.btn_apply_v2_all.clicked.connect(lambda: self._apply_voter_all(2))
        btn_layout.addWidget(self.btn_apply_v2_all)

        self.btn_apply_v3_all = QPushButton("Use All Voter 3 Values")
        self.btn_apply_v3_all.clicked.connect(lambda: self._apply_voter_all(3))
        btn_layout.addWidget(self.btn_apply_v3_all)

        btn_layout.addStretch()

        self.btn_approve = QPushButton("✅ Approve & Save Merged Record")
        self.btn_approve.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_approve.clicked.connect(self._approve_merged_record)
        btn_layout.addWidget(self.btn_approve)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _populate_diff_table(self) -> None:
        """Populate field rows comparing Voter 1, 2, 3 extractions vs Final Record."""
        fields: list[FieldSpec] = [
            ("event_timestamp", "Event Date", lambda r: r.event_timestamp.isoformat() if r.event_timestamp else ""),
            ("asset_type", "Asset Type", lambda r: str(r.asset_type or "")),
            ("symbol", "Symbol (Ticker)", lambda r: str(r.symbol or "")),
            ("isin", "ISIN Code", lambda r: str(r.isin or "")),
            ("action", "Action", lambda r: str(r.action or "")),
            ("quantity", "Quantity", lambda r: str(r.quantity) if r.quantity is not None else ""),
            ("unit_price", "Unit Price", lambda r: str(r.unit_price) if r.unit_price is not None else ""),
            ("currency", "Currency", lambda r: str(r.currency or "EUR")),
            ("fees", "Fees", lambda r: str(r.fees) if r.fees is not None else ""),
            ("total_amount", "Total Amount", lambda r: str(r.total_amount) if r.total_amount is not None else ""),
            ("fx_rate", "FX Rate", lambda r: str(r.fx_rate) if r.fx_rate is not None else ""),
            (
                "local_total_amount",
                "Local Total (EUR)",
                lambda r: str(r.local_total_amount) if r.local_total_amount is not None else "",
            ),
        ]

        def _extract_voter_dict(raw: object) -> dict[str, str]:
            if isinstance(raw, list):
                raw_list = cast(list[object], raw)
                for item in raw_list:
                    if isinstance(item, dict):
                        item_dict = cast(dict[object, object], item)
                        return {str(k): str(v) if v is not None else "" for k, v in item_dict.items()}
            return {}

        v1_data = _extract_voter_dict(self.consensus_log_data.get("raw_voter_1_records"))
        v2_data = _extract_voter_dict(self.consensus_log_data.get("raw_voter_2_records"))
        v3_data = _extract_voter_dict(self.consensus_log_data.get("raw_voter_3_records"))

        def _norm(key: str, val_str: str | None) -> str:
            if not val_str:
                return ""
            if key in ("quantity", "unit_price", "fees", "total_amount", "fx_rate", "local_total_amount"):
                try:
                    d = Decimal(val_str.strip().replace(",", "."))
                    norm_d = d.normalize()
                    # Format float without scientific notation
                    formatted = f"{norm_d:f}"
                    return formatted
                except Exception:
                    return val_str.strip()
            return val_str.strip()

        def _is_equivalent(key: str, s1: str, s2: str) -> bool:
            if key in ("quantity", "unit_price", "fees", "total_amount", "fx_rate", "local_total_amount"):
                if not s1 and not s2:
                    return True
                try:
                    d1 = Decimal(s1) if s1 else Decimal("0")
                    d2 = Decimal(s2) if s2 else Decimal("0")
                    return d1 == d2
                except Exception:
                    return s1 == s2
            return s1.lower() == s2.lower()

        self.table.setRowCount(len(fields))

        for row, (field_key, field_name, getter) in enumerate(fields):
            # Column 0: Field Name
            item_field = QTableWidgetItem(field_name)
            item_field.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f_font = QFont()
            f_font.setBold(True)
            item_field.setFont(f_font)
            self.table.setItem(row, 0, item_field)

            raw_v1_val = v1_data.get(field_key, getter(self.record))
            raw_v2_val = v2_data.get(field_key, getter(self.record))
            raw_v3_val = v3_data.get(field_key, getter(self.record))
            raw_curr_val = getter(self.record)

            val_v1 = _norm(field_key, raw_v1_val)
            val_v2 = _norm(field_key, raw_v2_val)
            val_v3 = _norm(field_key, raw_v3_val)
            val_curr = _norm(field_key, raw_curr_val)

            # Check if all voters agree
            is_mismatch = not (_is_equivalent(field_key, val_v1, val_v2) and _is_equivalent(field_key, val_v2, val_v3))

            # Column 1: Voter 1
            item_v1 = QTableWidgetItem(val_v1)
            item_v1.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, item_v1)

            # Column 2: Voter 2
            item_v2 = QTableWidgetItem(val_v2)
            item_v2.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 2, item_v2)

            # Column 3: Voter 3
            item_v3 = QTableWidgetItem(val_v3)
            item_v3.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 3, item_v3)

            # Column 4: Final Merged Value (Editable)
            item_merged = QTableWidgetItem(val_curr)
            self.table.setItem(row, 4, item_merged)

            if is_mismatch:
                mismatch_color = QColor(254, 249, 231)  # Light amber highlight
                item_v1.setBackground(mismatch_color)
                item_v2.setBackground(mismatch_color)
                item_v3.setBackground(mismatch_color)
                item_merged.setBackground(QColor(253, 237, 236))  # Light red editable highlight

            def _make_setter(r: int, v: str) -> Callable[[object], None]:
                def _set(_: object) -> None:
                    it = self.table.item(r, 4)
                    if it:
                        it.setText(v)

                return _set

            # Apply buttons per field row
            btn_v1 = QPushButton("← V1")
            btn_v1.clicked.connect(_make_setter(row, val_v1))
            self.table.setCellWidget(row, 5, btn_v1)

            btn_v2 = QPushButton("← V2")
            btn_v2.clicked.connect(_make_setter(row, val_v2))
            self.table.setCellWidget(row, 6, btn_v2)

            btn_v3 = QPushButton("← V3")
            btn_v3.clicked.connect(_make_setter(row, val_v3))
            self.table.setCellWidget(row, 7, btn_v3)

    def _apply_voter_all(self, voter_idx: int) -> None:
        """Apply all values from a specific voter (1, 2, or 3) into Final Merged column."""
        col_src = voter_idx  # col 1, 2, or 3
        for row in range(self.table.rowCount()):
            src_item = self.table.item(row, col_src)
            tgt_item = self.table.item(row, 4)
            if src_item and tgt_item:
                tgt_item.setText(src_item.text())

    def _approve_merged_record(self) -> None:
        """Collect merged table fields, update record, and call DatabaseManager to approve."""
        try:
            # Column 4 field indexing order matching `fields` array
            field_map = {
                0: "event_timestamp",
                1: "asset_type",
                2: "symbol",
                3: "isin",
                4: "action",
                5: "quantity",
                6: "unit_price",
                7: "currency",
                8: "fees",
                9: "total_amount",
                10: "fx_rate",
                11: "local_total_amount",
            }

            for row in range(self.table.rowCount()):
                f_name = field_map[row]
                it = self.table.item(row, 4)
                text_val = it.text().strip() if it else ""

                if f_name == "event_timestamp":
                    if text_val:
                        from datetime import datetime

                        setattr(self.record, f_name, datetime.fromisoformat(text_val))
                elif f_name in ("quantity", "unit_price", "fees", "total_amount", "fx_rate", "local_total_amount"):
                    if text_val:
                        setattr(self.record, f_name, abs(Decimal(text_val)))
                    else:
                        setattr(self.record, f_name, None)
                else:
                    setattr(self.record, f_name, text_val if text_val else None)

            # Update record in database
            if isinstance(self.record, StagedFinancialRecord):
                if self.record.verification_status == "escalated_to_user":
                    self.record.verification_status = "pending_approval"
                self.db.update_staged_record(self.record)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Staged record #{self.record.id} successfully updated! You can now approve it into the official ledger.",
                )
            else:
                self.record.verification_status = "approved"
                strict_record = BaseStrictRecord.from_raw(self.record)
                self.db.update_strict_financial_record(strict_record)
                QMessageBox.information(self, "Success", f"Record #{self.record.id} successfully updated and approved!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save record changes: {e}")

