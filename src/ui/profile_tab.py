"""Taxpayer Profile UI tab for Tax Return AI.

Allows users to manage their tax residency and domicile status per tax year.
"""

from __future__ import annotations

from decimal import Decimal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sqlmodel import Session, col, select

from backend.db_manager import DatabaseManager
from src.jurisdiction.ireland.cgt_models import ResidencyType, TaxpayerProfile, infer_residency_type


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


class TaxpayerProfileTab(QWidget):
    """UI tab for managing tax residence and domicile status per tax year."""

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._profiles: list[TaxpayerProfile] = []
        self._selected_profile: TaxpayerProfile | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Header bar
        header = QHBoxLayout()
        header.addWidget(_section_label("Taxpayer Profile per Tax Year"))
        header.addStretch()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.load_profiles)
        header.addWidget(btn_refresh)
        root.addLayout(header)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setHandleWidth(4)

        # Top section: Table
        splitter.addWidget(self._build_table_section())

        # Bottom section: Form
        splitter.addWidget(self._build_form_section())

        splitter.setSizes([350, 350])
        root.addWidget(splitter)

        self.load_profiles()

    # ------------------------------------------------------------------
    # UI Component Builders
    # ------------------------------------------------------------------

    def _build_table_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            [
                "Tax Year",
                "Fiscal Residence",
                "Domicile Country",
                "Is Domiciled in IE?",
                "Residency Type",
                "Notes",
            ]
        )

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        layout.addWidget(self._table)
        return panel

    def _build_form_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(_section_label("Add / Edit Taxpayer Profile"))
        layout.addWidget(_divider())

        form = QFormLayout()
        form.setSpacing(8)

        self._input_year = QLineEdit()
        self._input_year.setPlaceholderText("e.g. 2025")
        form.addRow("Tax Year:", self._input_year)

        self._combo_residence = QComboBox()
        self._combo_residence.setEditable(True)
        self._combo_residence.addItems(["IE", "IT", "US", "GB"])
        self._combo_residence.currentTextChanged.connect(self._auto_infer_residency_type)
        form.addRow("Fiscal Residence Country:", self._combo_residence)

        self._combo_domicile = QComboBox()
        self._combo_domicile.setEditable(True)
        self._combo_domicile.addItems(["IT", "IE", "US", "GB"])
        self._combo_domicile.currentTextChanged.connect(self._auto_infer_residency_type)
        form.addRow("Domicile Country:", self._combo_domicile)

        # Residency type row with override checkbox
        residency_row = QHBoxLayout()
        self._combo_residency_type = QComboBox()
        self._combo_residency_type.setEnabled(False)  # Read-only by default (auto-inferred)
        for r in ResidencyType:
            self._combo_residency_type.addItem(r.value, r)
        residency_row.addWidget(self._combo_residency_type, stretch=1)

        self._chk_override = QCheckBox("Manual Override")
        self._chk_override.toggled.connect(self._combo_residency_type.setEnabled)
        residency_row.addWidget(self._chk_override)

        form.addRow("Residency Type:", residency_row)

        # Tax implication explanation label
        self._lbl_explanation = QLabel()
        self._lbl_explanation.setStyleSheet("color: #4A5568; font-style: italic; font-size: 11px;")
        form.addRow("Tax Status:", self._lbl_explanation)

        # Marginal Tax Rate row
        self._input_marginal_rate = QLineEdit()
        self._input_marginal_rate.setPlaceholderText("e.g. 0.40 or 0.52")
        self._input_marginal_rate.setText("0.40")
        form.addRow("Marginal Tax Rate:", self._input_marginal_rate)

        self._input_notes = QLineEdit()
        self._input_notes.setPlaceholderText("Optional notes")
        form.addRow("Notes:", self._input_notes)

        # Trigger initial auto-inference
        self._auto_infer_residency_type()

        layout.addLayout(form)

        # Action Buttons
        btn_bar = QHBoxLayout()
        btn_save = QPushButton("Save Profile")
        btn_save.clicked.connect(self._on_save_clicked)
        btn_bar.addWidget(btn_save)

        btn_clear = QPushButton("Clear Form")
        btn_clear.clicked.connect(self._clear_form)
        btn_bar.addWidget(btn_clear)

        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        return panel

    # ------------------------------------------------------------------
    # Data Operations
    # ------------------------------------------------------------------

    def load_profiles(self) -> None:
        """Load all taxpayer profiles from database into table."""
        self._profiles = []
        with Session(self._db.engine) as session:
            statement = select(TaxpayerProfile).order_by(col(TaxpayerProfile.tax_year))
            self._profiles = list(session.exec(statement).all())

        self._table.setRowCount(len(self._profiles))
        for row, p in enumerate(self._profiles):
            self._table.setItem(row, 0, QTableWidgetItem(str(p.tax_year)))
            self._table.setItem(row, 1, QTableWidgetItem(p.fiscal_residence_country))
            self._table.setItem(row, 2, QTableWidgetItem(p.domicile_country))
            self._table.setItem(row, 3, QTableWidgetItem("Yes" if p.is_domiciled_in_ireland else "No"))
            self._table.setItem(row, 4, QTableWidgetItem(p.residency_type))
            self._table.setItem(row, 5, QTableWidgetItem(p.notes or ""))

    def _on_row_selected(self) -> None:
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            self._selected_profile = None
            return

        row = selected_rows[0].row()
        if row < len(self._profiles):
            p = self._profiles[row]
            self._selected_profile = p
            self._input_year.setText(str(p.tax_year))
            self._combo_residence.setEditText(p.fiscal_residence_country)
            self._combo_domicile.setEditText(p.domicile_country)
            self._input_marginal_rate.setText(str(p.marginal_tax_rate))
            self._input_notes.setText(p.notes or "")

            idx = self._combo_residency_type.findText(p.residency_type)
            if idx >= 0:
                self._combo_residency_type.setCurrentIndex(idx)

    def _auto_infer_residency_type(self) -> None:
        """Automatically infer residency type based on selected residence & domicile countries."""
        residence = self._combo_residence.currentText().strip().upper()
        domicile = self._combo_domicile.currentText().strip().upper()

        if residence and domicile:
            inferred = infer_residency_type(residence, domicile)
            if not self._chk_override.isChecked():
                idx = self._combo_residency_type.findText(inferred.value)
                if idx >= 0:
                    self._combo_residency_type.setCurrentIndex(idx)

            # Update status explanation label
            if inferred == ResidencyType.RESIDENT_DOMICILED:
                self._lbl_explanation.setText(
                    "Irish Resident & Domiciled -> Taxable in Ireland on worldwide capital gains."
                )
            elif inferred == ResidencyType.RESIDENT_NON_DOMICILED:
                self._lbl_explanation.setText(
                    "Irish Resident, Non-Domiciled -> Taxable on Irish gains; foreign gains taxable ONLY if remitted to Ireland."
                )
            else:
                self._lbl_explanation.setText("Non-Resident in Ireland -> Not subject to Irish CGT on foreign assets.")

    def _clear_form(self) -> None:
        self._selected_profile = None
        self._input_year.clear()
        self._combo_residence.setCurrentIndex(0)
        self._combo_domicile.setCurrentIndex(0)
        self._chk_override.setChecked(False)
        self._input_marginal_rate.setText("0.40")
        self._input_notes.clear()
        self._auto_infer_residency_type()
        self._table.clearSelection()

    def _on_save_clicked(self) -> None:
        year_str = self._input_year.text().strip()
        residence = self._combo_residence.currentText().strip().upper()
        domicile = self._combo_domicile.currentText().strip().upper()
        residency_type = self._combo_residency_type.currentText()
        rate_str = self._input_marginal_rate.text().strip()
        notes = self._input_notes.text().strip() or None

        if not year_str.isdigit():
            QMessageBox.warning(self, "Validation Error", "Tax year must be an integer.")
            return
        if not residence or len(residence) != 2:
            QMessageBox.warning(self, "Validation Error", "Residence country must be a 2-letter ISO code.")
            return
        if not domicile or len(domicile) != 2:
            QMessageBox.warning(self, "Validation Error", "Domicile country must be a 2-letter ISO code.")
            return

        try:
            marginal_tax_rate = Decimal(rate_str)
        except Exception:
            QMessageBox.warning(self, "Validation Error", "Marginal tax rate must be a valid Decimal (e.g. 0.40).")
            return

        tax_year = int(year_str)

        profile = TaxpayerProfile(
            id=self._selected_profile.id if self._selected_profile else None,
            tax_year=tax_year,
            fiscal_residence_country=residence,
            domicile_country=domicile,
            residency_type=residency_type,
            marginal_tax_rate=marginal_tax_rate,
            notes=notes,
        )

        try:
            self._db.upsert_taxpayer_profile(profile)
            QMessageBox.information(self, "Success", f"Saved taxpayer profile for tax year {tax_year}.")
            self._clear_form()
            self.load_profiles()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save profile: {exc}")
