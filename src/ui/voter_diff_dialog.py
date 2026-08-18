"""Reusable side-by-side 3-Voter Diff & Merge Dialog for PySide6 UI."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VoterDiffField:
    """Field specification for a single row in the 3-voter comparison dialog."""

    key: str
    label: str
    val_v1: str = ""
    val_v2: str = ""
    val_v3: str = ""
    initial_merged_val: str = ""
    is_numeric: bool = False


class GenericVoterDiffDialog(QDialog):
    """Reusable 3-Voter Diff and Merge Dialog (pure presentation & user selection).

    Renders a side-by-side 8-column comparison table (Field | Voter 1 | Voter 2 | Voter 3 |
    Final Merged | ← V1 | ← V2 | ← V3), highlights mismatches, and outputs a dictionary of
    merged field values upon acceptance.
    """

    def __init__(
        self,
        title: str,
        header_html: str,
        fields: list[VoterDiffField],
        accept_button_text: str = "✅ Apply & Merge",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.fields: list[VoterDiffField] = fields
        self._merged_values: dict[str, str] = {}

        self.setWindowTitle(title)
        self.resize(1100, 650)
        self._init_ui(header_html, accept_button_text)

    def _init_ui(self, header_html: str, accept_button_text: str) -> None:
        layout = QVBoxLayout(self)

        info_label = QLabel(header_html)
        info_label.setStyleSheet("font-size: 13px; margin-bottom: 8px;")
        layout.addWidget(info_label)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            [
                "Field",
                "Voter 1 Extraction",
                "Voter 2 Extraction",
                "Voter 3 Extraction",
                "Final Merged Value (Editable)",
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

        self.btn_accept = QPushButton(accept_button_text)
        self.btn_accept.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_accept.clicked.connect(self._on_accept_clicked)
        btn_layout.addWidget(self.btn_accept)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    @staticmethod
    def is_numeric_equiv(s1: str, s2: str) -> bool:
        """Check if two string values are numerically equivalent."""
        if not s1 and not s2:
            return True
        try:
            d1 = Decimal(s1.strip().replace(",", "."))
            d2 = Decimal(s2.strip().replace(",", "."))
            return d1 == d2
        except Exception:
            return s1.strip() == s2.strip()

    def _populate_diff_table(self) -> None:
        """Populate rows comparing Voter 1, 2, 3 extractions vs Final Record."""
        self.table.setRowCount(len(self.fields))

        for row, spec in enumerate(self.fields):
            item_field = QTableWidgetItem(spec.label)
            item_field.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f_font = QFont()
            f_font.setBold(True)
            item_field.setFont(f_font)
            self.table.setItem(row, 0, item_field)

            v1_clean = spec.val_v1.strip()
            v2_clean = spec.val_v2.strip()
            v3_clean = spec.val_v3.strip()
            curr_clean = spec.initial_merged_val.strip()

            if spec.is_numeric:
                is_mismatch = not (
                    self.is_numeric_equiv(v1_clean, v2_clean) and self.is_numeric_equiv(v2_clean, v3_clean)
                )
            else:
                is_mismatch = not (v1_clean.lower() == v2_clean.lower() and v2_clean.lower() == v3_clean.lower())

            item_v1 = QTableWidgetItem(v1_clean)
            item_v1.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, item_v1)

            item_v2 = QTableWidgetItem(v2_clean)
            item_v2.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 2, item_v2)

            item_v3 = QTableWidgetItem(v3_clean)
            item_v3.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 3, item_v3)

            item_merged = QTableWidgetItem(curr_clean)
            self.table.setItem(row, 4, item_merged)

            if is_mismatch or not curr_clean:
                mismatch_color = QColor(254, 249, 231)  # Light amber
                item_v1.setBackground(mismatch_color)
                item_v2.setBackground(mismatch_color)
                item_v3.setBackground(mismatch_color)
                item_merged.setBackground(QColor(253, 237, 236))  # Soft pink

            def _make_setter(r: int, v: str) -> Callable[[object], None]:
                def _set(_: object) -> None:
                    it = self.table.item(r, 4)
                    if it:
                        it.setText(v)

                return _set

            btn_v1 = QPushButton("← V1")
            btn_v1.clicked.connect(_make_setter(row, v1_clean))
            self.table.setCellWidget(row, 5, btn_v1)

            btn_v2 = QPushButton("← V2")
            btn_v2.clicked.connect(_make_setter(row, v2_clean))
            self.table.setCellWidget(row, 6, btn_v2)

            btn_v3 = QPushButton("← V3")
            btn_v3.clicked.connect(_make_setter(row, v3_clean))
            self.table.setCellWidget(row, 7, btn_v3)

    def _apply_voter_all(self, voter_idx: int) -> None:
        """Apply all values from a specific voter (1, 2, or 3) into Final Merged column."""
        for row in range(self.table.rowCount()):
            src_item = self.table.item(row, voter_idx)
            tgt_item = self.table.item(row, 4)
            if src_item and tgt_item:
                tgt_item.setText(src_item.text())

    def _on_accept_clicked(self) -> None:
        """Extract merged field values from table and accept dialog."""
        self._merged_values = {}
        for row, spec in enumerate(self.fields):
            it = self.table.item(row, 4)
            self._merged_values[spec.key] = it.text().strip() if it else ""
        self.accept()

    def get_merged_values(self) -> dict[str, str]:
        """Return the user-selected and edited merged field dictionary."""
        return dict(self._merged_values)


class VoterDiffMergeDialog(GenericVoterDiffDialog):
    """Interactive 3-Voter Diff and Merge Dialog for Financial Transactions.

    Populates voter extraction fields from consensus_log, allows user to review and
    resolve differences, and provides helper to persist the merged record into the database.
    """

    def __init__(
        self, record: FinancialRecord | StagedFinancialRecord, db: DatabaseManager, parent: QWidget | None = None
    ) -> None:
        self.record: FinancialRecord | StagedFinancialRecord = record
        self.db: DatabaseManager = db

        consensus_log_data: dict[str, object] = {}
        if record.consensus_log:
            try:
                parsed = json.loads(record.consensus_log)
                if isinstance(parsed, dict):
                    consensus_log_data = cast(dict[str, object], parsed)
            except Exception:
                consensus_log_data = {}


        header_text = (
            f"<b>Source File:</b> {record.source_file_name or 'N/A'} | "
            f"<b>Provider:</b> {record.provider or 'N/A'} | "
            f"<b>Status:</b> <font color='#e67e22'>{record.verification_status.upper()}</font>"
        )
        if consensus_log_data.get("error"):
            header_text += (
                f"<br/><font color='#c0392b'><b>Consensus Alert:</b> {consensus_log_data.get('error')}</font>"
            )

        fields = self._build_field_specs(record, consensus_log_data)

        super().__init__(
            title=f"🏛️ Voter Consensus Diff & Merge - Record #{record.id} ({record.source_file_name or 'Document'})",
            header_html=header_text,
            fields=fields,
            accept_button_text="✅ Approve & Save Merged Record",
            parent=parent,
        )

    @staticmethod
    def _build_field_specs(
        record: FinancialRecord | StagedFinancialRecord,
        consensus_log_data: dict[str, object],
    ) -> list[VoterDiffField]:
        specs: list[tuple[str, str, Callable[[FinancialRecord | StagedFinancialRecord], str], bool]] = [
            (
                "event_timestamp",
                "Event Date",
                lambda r: r.event_timestamp.isoformat() if r.event_timestamp else "",
                False,
            ),
            ("asset_type", "Asset Type", lambda r: str(r.asset_type or ""), False),
            ("symbol", "Symbol (Ticker)", lambda r: str(r.symbol or ""), False),
            ("isin", "ISIN Code", lambda r: str(r.isin or ""), False),
            ("action", "Action", lambda r: str(r.action or ""), False),
            ("quantity", "Quantity", lambda r: str(r.quantity) if r.quantity is not None else "", True),
            ("unit_price", "Unit Price", lambda r: str(r.unit_price) if r.unit_price is not None else "", True),
            ("currency", "Currency", lambda r: str(r.currency or "EUR"), False),
            ("fees", "Fees", lambda r: str(r.fees) if r.fees is not None else "", True),
            (
                "total_amount",
                "Total Amount",
                lambda r: str(r.total_amount) if r.total_amount is not None else "",
                True,
            ),
            ("fx_rate", "FX Rate", lambda r: str(r.fx_rate) if r.fx_rate is not None else "", True),
            (
                "local_total_amount",
                "Local Total (EUR)",
                lambda r: str(r.local_total_amount) if r.local_total_amount is not None else "",
                True,
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

        v1_data = _extract_voter_dict(consensus_log_data.get("raw_voter_1_records"))
        v2_data = _extract_voter_dict(consensus_log_data.get("raw_voter_2_records"))
        v3_data = _extract_voter_dict(consensus_log_data.get("raw_voter_3_records"))

        result: list[VoterDiffField] = []
        for key, label, getter, is_num in specs:
            raw_v1 = v1_data.get(key, getter(record))
            raw_v2 = v2_data.get(key, getter(record))
            raw_v3 = v3_data.get(key, getter(record))
            curr = getter(record)
            result.append(
                VoterDiffField(
                    key=key,
                    label=label,
                    val_v1=raw_v1,
                    val_v2=raw_v2,
                    val_v3=raw_v3,
                    initial_merged_val=curr,
                    is_numeric=is_num,
                )
            )
        return result

    def _on_accept_clicked(self) -> None:
        """Override to save merged record upon clicking accept."""
        try:
            self._merged_values = {}
            for row, spec in enumerate(self.fields):
                it = self.table.item(row, 4)
                self._merged_values[spec.key] = it.text().strip() if it else ""

            self._save_merged_record(self._merged_values)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save record changes: {e}")

    def _approve_merged_record(self) -> None:
        """Compatibility helper for saving and approving merged record."""
        self._on_accept_clicked()

    def _save_merged_record(self, merged: dict[str, str]) -> None:
        for f_name, text_val in merged.items():
            if f_name == "event_timestamp":
                if text_val:
                    from datetime import datetime

                    setattr(self.record, f_name, datetime.fromisoformat(text_val))
            elif f_name in ("quantity", "unit_price", "fees", "total_amount", "fx_rate", "local_total_amount"):
                if text_val:
                    setattr(self.record, f_name, abs(Decimal(text_val.replace(",", "."))))
                else:
                    setattr(self.record, f_name, None)
            else:
                setattr(self.record, f_name, text_val if text_val else None)

        if isinstance(self.record, StagedFinancialRecord):
            if self.record.verification_status == "escalated_to_user":
                self.record.verification_status = "pending_approval"
            self.db.update_staged_record(self.record)
            QMessageBox.information(
                self,
                "Success",
                f"Staged record #{self.record.id} successfully updated! "
                "You can now approve it into the official ledger.",
            )

        else:
            self.record.verification_status = "approved"
            strict_record = BaseStrictRecord.from_raw(self.record)
            self.db.update_strict_financial_record(strict_record)
            QMessageBox.information(self, "Success", f"Record #{self.record.id} successfully updated and approved!")
