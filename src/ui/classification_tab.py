"""Asset Classification UI tab for Tax Return AI.

Allows users to manage ISIN tax regimes, asset categories, and UCITS flags for Irish tax logic.
Also includes OpenFIGI auto-enrichment and dirty form change protection.
"""

from __future__ import annotations

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
from sqlmodel import Session, select

from backend.ingestion.openfigi import OpenFIGIMapper
from src.jurisdiction.ireland.cgt_models import (
    AssetTaxClassification,
    AssetTaxClassificationDomain,
    IrishTaxRegime,
    infer_tax_regime,
    parse_irish_tax_regime,
)
from src.ui.config import UIConfig

# Descriptions of tax rules for each Irish regime
REGIME_EXPLANATIONS: dict[IrishTaxRegime, str] = {
    IrishTaxRegime.EXIT_TAX: (
        "Exit Tax (41% pre-2026, 38% 2026+) -> Losses trapped (no offset), no €1,270 exemption, 8-yr deemed disposal."
    ),
    IrishTaxRegime.CGT_STANDARD: (
        "Standard CGT (33%) -> FIFO matching (Sec 580), Sec 581 loss quarantine, loss offset allowed, €1,270 annual exemption."
    ),
    IrishTaxRegime.OFFSHORE_DISTRIBUTING: (
        "Offshore Distributing Fund (40%) -> Loss offset allowed in CGT pool, €1,270 annual exemption applicable."
    ),
    IrishTaxRegime.OFFSHORE_NON_DISTRIBUTING: (
        "Offshore Non-Distributing Fund (Income Tax up to 52%) -> Disallowed vs CGT losses, no annual exemption."
    ),
    IrishTaxRegime.ETC_COMMODITY: (
        "Exchange Traded Commodity (33%) -> Taxed as debt security/CGT, loss offset allowed, €1,270 annual exemption."
    ),
}

# Supported Asset Category options for UI dropdown
CATEGORY_DIRECT_EQUITY = "Direct Equity / Crypto / Bond (CGT)"
CATEGORY_UCITS_ETF = "EU UCITS ETF / Fund (Exit Tax)"
CATEGORY_ETC = "Exchange Traded Commodity / ETC (CGT)"
CATEGORY_OFFSHORE_DIST = "Offshore Distributing Fund (40%)"
CATEGORY_OFFSHORE_NON_DIST = "Offshore Non-Distributing Fund (Income Tax)"

ASSET_CATEGORIES: list[str] = [
    CATEGORY_DIRECT_EQUITY,
    CATEGORY_UCITS_ETF,
    CATEGORY_ETC,
    CATEGORY_OFFSHORE_DIST,
    CATEGORY_OFFSHORE_NON_DIST,
]


def _section_label(text: str) -> QLabel:
    """Create a bold section label."""
    lbl = QLabel(text)
    font = lbl.font()
    font.setBold(True)
    lbl.setFont(font)
    return lbl


def _divider() -> QFrame:
    """Create a horizontal sunken divider line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _detect_asset_traits_from_metadata(
    isin: str,
    asset_name: str,
) -> tuple[str, str | None]:
    """Detect asset category and domicile country from ISIN and asset name heuristics.

    Args:
        isin: 12-character ISIN code.
        asset_name: Descriptive asset name from OpenFIGI or manual input.

    Returns:
        Tuple of (detected_category_label, detected_domicile_code_or_none).
    """
    name_upper = asset_name.upper()
    category = CATEGORY_DIRECT_EQUITY

    if "UCITS" in name_upper:
        category = CATEGORY_UCITS_ETF
    elif any(kw in name_upper for kw in (" ETC", "COMMODITY", "PHYSICAL GOLD", "PHYSICAL SILVER")):
        category = CATEGORY_ETC
    elif "OFFSHORE DISTRIBUTING" in name_upper:
        category = CATEGORY_OFFSHORE_DIST
    elif "OFFSHORE NON-DISTRIBUTING" in name_upper or "NON-DISTRIBUTING" in name_upper:
        category = CATEGORY_OFFSHORE_NON_DIST

    domicile: str | None = None
    prefix = isin[:2].upper()
    if prefix in ("IE", "LU", "US", "DE", "FR", "GB", "IT", "CH", "NL", "JE", "GG"):
        domicile = prefix

    return category, domicile


class AssetClassificationTab(QWidget):
    """UI tab for managing ISIN tax classification, UCITS status, and OpenFIGI enrichment."""

    def __init__(self, config: UIConfig, parent: QWidget | None = None) -> None:
        """Initialize the Asset Classification tab.

        Args:
            config: Centralized UIConfig instance containing database manager.
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self._config = config
        self._db = config.db
        self._figi_mapper = OpenFIGIMapper()
        self._classifications: list[AssetTaxClassificationDomain] = []
        self._selected_item: AssetTaxClassificationDomain | None = None
        self._selected_row_idx: int | None = None
        self._is_updating_selection: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Header bar
        header = QHBoxLayout()
        header.addWidget(_section_label("Asset Tax Classification (ISIN / UCITS / Regimes)"))
        header.addStretch()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.load_classifications)
        header.addWidget(btn_refresh)
        root.addLayout(header)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setHandleWidth(4)

        # Top section: Table
        splitter.addWidget(self._build_table_section())

        # Bottom section: Form
        splitter.addWidget(self._build_form_section())

        splitter.setSizes([360, 340])
        root.addWidget(splitter)

        self.load_classifications()

    # ------------------------------------------------------------------
    # UI Component Builders
    # ------------------------------------------------------------------

    def _build_table_section(self) -> QWidget:
        """Build table section displaying asset classifications."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            [
                "ISIN",
                "Asset Name",
                "Tax Regime",
                "Domicile",
                "UCITS?",
                "Source",
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
        """Build form section for adding and editing asset classifications."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        header_box = QHBoxLayout()
        header_box.addWidget(_section_label("Asset Classification Details"))
        header_box.addStretch()
        self._lbl_mode = QLabel("Mode: Adding New Asset")
        self._lbl_mode.setStyleSheet("font-weight: bold; color: #2B6CB0; font-size: 12px;")
        header_box.addWidget(self._lbl_mode)
        layout.addLayout(header_box)

        layout.addWidget(_divider())

        form = QFormLayout()
        form.setSpacing(8)

        # ISIN row + OpenFIGI button
        isin_box = QHBoxLayout()
        self._input_isin = QLineEdit()
        self._input_isin.setPlaceholderText("e.g. IE00BFWXDV39")
        isin_box.addWidget(self._input_isin)

        btn_figi = QPushButton("Auto-Fill via OpenFIGI")
        btn_figi.clicked.connect(self._on_autofill_clicked)
        isin_box.addWidget(btn_figi)

        form.addRow("ISIN Code:", isin_box)

        self._input_name = QLineEdit()
        self._input_name.setPlaceholderText("e.g. Vanguard S&P 500 UCITS ETF")
        form.addRow("Asset Name:", self._input_name)

        # Asset Category selector (drives auto-inference)
        self._combo_category = QComboBox()
        self._combo_category.addItems(ASSET_CATEGORIES)
        self._combo_category.currentIndexChanged.connect(self._on_category_changed)
        form.addRow("Asset Category:", self._combo_category)

        # Inferred regime display & manual override toggle
        regime_control_box = QHBoxLayout()
        self._combo_regime = QComboBox()
        for regime in IrishTaxRegime:
            self._combo_regime.addItem(regime.value, regime)
        self._combo_regime.currentTextChanged.connect(self._update_regime_explanation)
        self._combo_regime.setEnabled(False)  # Locked by default
        regime_control_box.addWidget(self._combo_regime)

        self._chk_override_regime = QCheckBox("Manual Override")
        self._chk_override_regime.toggled.connect(self._on_override_toggled)
        regime_control_box.addWidget(self._chk_override_regime)

        self._lbl_inferred_regime = QLabel("Inferred: cgt_standard")
        self._lbl_inferred_regime.setStyleSheet("color: #718096; font-size: 11px; font-style: italic;")
        regime_control_box.addWidget(self._lbl_inferred_regime)
        regime_control_box.addStretch()

        form.addRow("Tax Regime:", regime_control_box)

        self._lbl_regime_explanation = QLabel()
        self._lbl_regime_explanation.setWordWrap(True)
        self._lbl_regime_explanation.setStyleSheet("color: #4A5568; font-style: italic; font-size: 11px;")
        form.addRow("Regime Rules:", self._lbl_regime_explanation)

        self._combo_domicile = QComboBox()
        self._combo_domicile.setEditable(True)
        self._combo_domicile.addItems(["IE", "LU", "US", "IT", "DE", "FR", "GB", "CH", "NL"])
        form.addRow("Domicile Country:", self._combo_domicile)

        self._input_notes = QLineEdit()
        self._input_notes.setPlaceholderText("Optional notes")
        form.addRow("Notes:", self._input_notes)

        # Initial explanation
        self._auto_infer_tax_regime()

        layout.addLayout(form)

        # Action Buttons
        btn_bar = QHBoxLayout()
        btn_save = QPushButton("Save Classification")
        btn_save.setStyleSheet("font-weight: bold;")
        btn_save.clicked.connect(self._on_save_clicked)
        btn_bar.addWidget(btn_save)

        btn_deselect = QPushButton("Deselect / Add New")
        btn_deselect.clicked.connect(self._on_deselect_new_clicked)
        btn_bar.addWidget(btn_deselect)

        btn_clear = QPushButton("Clear Form")
        btn_clear.clicked.connect(self._on_clear_clicked)
        btn_bar.addWidget(btn_clear)

        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        return panel

    # ------------------------------------------------------------------
    # Data Operations
    # ------------------------------------------------------------------

    def load_classifications(self) -> None:
        """Load all asset classifications from database into domain models and table."""
        self._classifications = []
        with Session(self._db.engine) as session:
            statement = select(AssetTaxClassification).order_by(AssetTaxClassification.isin)
            db_records = list(session.exec(statement).all())
            self._classifications = [r.to_domain() for r in db_records]

        self._table.setRowCount(len(self._classifications))
        for row, c in enumerate(self._classifications):
            self._table.setItem(row, 0, QTableWidgetItem(c.isin))
            self._table.setItem(row, 1, QTableWidgetItem(c.asset_name or ""))
            regime_str = c.tax_regime.value
            self._table.setItem(row, 2, QTableWidgetItem(regime_str))
            self._table.setItem(row, 3, QTableWidgetItem(c.domicile_country or ""))
            self._table.setItem(row, 4, QTableWidgetItem("Yes" if c.is_ucits else "No"))
            self._table.setItem(row, 5, QTableWidgetItem(c.classification_source or "manual"))
            self._table.setItem(row, 6, QTableWidgetItem(c.notes or ""))

    # ------------------------------------------------------------------
    # Dirty Form & Change Tracking
    # ------------------------------------------------------------------

    def _is_form_dirty(self) -> bool:
        """Check if user has uncommitted modifications in the form compared to selected item.

        Returns:
            True if current form data differs from baseline state.
        """
        current_isin = self._input_isin.text().strip().upper()
        current_name = self._input_name.text().strip()
        current_domicile = self._combo_domicile.currentText().strip().upper()
        current_notes = self._input_notes.text().strip()
        current_category = self._combo_category.currentText()
        current_override = self._chk_override_regime.isChecked()
        current_regime = self._combo_regime.currentText()

        if self._selected_item is None:
            # Baseline: empty form
            return bool(
                current_isin
                or current_name
                or current_notes
                or current_override
                or current_category != CATEGORY_DIRECT_EQUITY
                or current_domicile != "IE"
            )

        # Baseline: selected item
        expected_category = self._category_from_domain(self._selected_item)
        expected_isin = self._selected_item.isin
        expected_name = self._selected_item.asset_name or ""
        expected_domicile = (self._selected_item.domicile_country or "").upper()
        expected_notes = self._selected_item.notes or ""
        expected_regime = self._selected_item.tax_regime.value

        inferred_regime = self._get_inferred_regime_for_category(expected_category).value
        expected_override = expected_regime != inferred_regime

        return (
            current_isin != expected_isin
            or current_name != expected_name
            or current_domicile != expected_domicile
            or current_notes != expected_notes
            or current_category != expected_category
            or current_override != expected_override
            or (current_override and current_regime != expected_regime)
        )

    def _confirm_discard_if_dirty(self) -> bool:
        """Prompt confirmation if form has unsaved modifications.

        Returns:
            True if safe to discard changes or form is not dirty; False if user cancelled.
        """
        if not self._is_form_dirty():
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes in the form. Do you want to discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------
    # Selection & State Management
    # ------------------------------------------------------------------

    def _category_from_domain(self, item: AssetTaxClassificationDomain) -> str:
        """Determine asset category string from domain model flags."""
        if item.is_ucits or item.tax_regime == IrishTaxRegime.EXIT_TAX:
            return CATEGORY_UCITS_ETF
        if item.is_etc or item.tax_regime == IrishTaxRegime.ETC_COMMODITY:
            return CATEGORY_ETC
        if item.is_offshore_distributing or item.tax_regime == IrishTaxRegime.OFFSHORE_DISTRIBUTING:
            return CATEGORY_OFFSHORE_DIST
        if item.tax_regime == IrishTaxRegime.OFFSHORE_NON_DISTRIBUTING:
            return CATEGORY_OFFSHORE_NON_DIST
        return CATEGORY_DIRECT_EQUITY

    def _on_row_selected(self) -> None:
        """Handle selection change in classifications table with dirty-check protection."""
        if self._is_updating_selection:
            return

        selected_rows = self._table.selectionModel().selectedRows()
        new_row_idx = selected_rows[0].row() if selected_rows else None

        if new_row_idx == self._selected_row_idx:
            return

        if not self._confirm_discard_if_dirty():
            # User cancelled: revert table selection
            self._is_updating_selection = True
            if self._selected_row_idx is not None and self._selected_row_idx < self._table.rowCount():
                self._table.selectRow(self._selected_row_idx)
            else:
                self._table.clearSelection()
            self._is_updating_selection = False
            return

        self._selected_row_idx = new_row_idx
        if new_row_idx is not None and new_row_idx < len(self._classifications):
            item = self._classifications[new_row_idx]
            self._load_item_into_form(item)
        else:
            self._set_add_mode()

    def _load_item_into_form(self, item: AssetTaxClassificationDomain) -> None:
        """Populate form with selected domain model data in Edit Mode."""
        self._selected_item = item
        self._lbl_mode.setText(f"Mode: Editing ISIN [{item.isin}]")
        self._lbl_mode.setStyleSheet("font-weight: bold; color: #C05621; font-size: 12px;")

        self._input_isin.setText(item.isin)
        self._input_name.setText(item.asset_name or "")
        self._combo_domicile.setEditText(item.domicile_country or "IE")
        self._input_notes.setText(item.notes or "")

        cat = self._category_from_domain(item)
        cat_idx = self._combo_category.findText(cat)
        if cat_idx >= 0:
            self._combo_category.setCurrentIndex(cat_idx)

        inferred = self._get_inferred_regime_for_category(cat)
        self._lbl_inferred_regime.setText(f"Inferred: {inferred.value}")

        is_overridden = item.tax_regime != inferred
        self._chk_override_regime.setChecked(is_overridden)
        self._combo_regime.setEnabled(is_overridden)

        regime_idx = self._combo_regime.findText(item.tax_regime.value)
        if regime_idx >= 0:
            self._combo_regime.setCurrentIndex(regime_idx)

        self._update_regime_explanation()

    def _set_add_mode(self) -> None:
        """Reset form controls and state to Add Mode."""
        self._selected_item = None
        self._selected_row_idx = None
        self._lbl_mode.setText("Mode: Adding New Asset")
        self._lbl_mode.setStyleSheet("font-weight: bold; color: #2B6CB0; font-size: 12px;")

        self._input_isin.clear()
        self._input_name.clear()
        self._combo_domicile.setCurrentIndex(0)
        self._input_notes.clear()
        self._combo_category.setCurrentIndex(0)
        self._chk_override_regime.setChecked(False)
        self._combo_regime.setEnabled(False)
        self._auto_infer_tax_regime()

    def _on_deselect_new_clicked(self) -> None:
        """Deselect any active table selection and switch to Add Mode after dirty check."""
        if not self._confirm_discard_if_dirty():
            return

        self._is_updating_selection = True
        self._table.clearSelection()
        self._is_updating_selection = False
        self._set_add_mode()

    def _on_clear_clicked(self) -> None:
        """Clear form inputs after dirty confirmation."""
        if not self._confirm_discard_if_dirty():
            return
        self._set_add_mode()

    # ------------------------------------------------------------------
    # Category, Override & Inference Logic
    # ------------------------------------------------------------------

    def _get_inferred_regime_for_category(self, category_text: str) -> IrishTaxRegime:
        """Infer IrishTaxRegime from category text selection.

        Args:
            category_text: Selected category string from dropdown.

        Returns:
            Inferred IrishTaxRegime enum value.
        """
        if category_text == CATEGORY_UCITS_ETF:
            return infer_tax_regime(
                is_ucits=True,
                is_etc=False,
                is_offshore_distributing=False,
                is_direct_equity_or_crypto=False,
            )
        if category_text == CATEGORY_ETC:
            return infer_tax_regime(
                is_ucits=False,
                is_etc=True,
                is_offshore_distributing=False,
                is_direct_equity_or_crypto=False,
            )
        if category_text == CATEGORY_OFFSHORE_DIST:
            return infer_tax_regime(
                is_ucits=False,
                is_etc=False,
                is_offshore_distributing=True,
                is_direct_equity_or_crypto=False,
            )
        if category_text == CATEGORY_OFFSHORE_NON_DIST:
            return IrishTaxRegime.OFFSHORE_NON_DISTRIBUTING

        return infer_tax_regime(
            is_ucits=False,
            is_etc=False,
            is_offshore_distributing=False,
            is_direct_equity_or_crypto=True,
        )

    def _on_category_changed(self) -> None:
        """Handle asset category dropdown changes."""
        self._auto_infer_tax_regime()

    def _auto_infer_tax_regime(self) -> None:
        """Automatically select tax regime based on chosen category unless manually overridden."""
        cat = self._combo_category.currentText()
        inferred = self._get_inferred_regime_for_category(cat)
        self._lbl_inferred_regime.setText(f"Inferred: {inferred.value}")

        if not self._chk_override_regime.isChecked():
            idx = self._combo_regime.findText(inferred.value)
            if idx >= 0:
                self._combo_regime.setCurrentIndex(idx)

        self._update_regime_explanation()

    def _on_override_toggled(self, checked: bool) -> None:
        """Toggle manual regime override mode.

        Args:
            checked: True if manual override is enabled.
        """
        self._combo_regime.setEnabled(checked)
        if not checked:
            self._auto_infer_tax_regime()
        else:
            self._update_regime_explanation()

    def _update_regime_explanation(self) -> None:
        """Update text label describing the selected tax regime rules via dictionary mapping."""
        regime_val = self._combo_regime.currentText()
        try:
            regime = IrishTaxRegime(regime_val)
            explanation = REGIME_EXPLANATIONS.get(regime, "")
        except ValueError:
            explanation = ""
        self._lbl_regime_explanation.setText(explanation)

    # ------------------------------------------------------------------
    # Autofill & Save Actions
    # ------------------------------------------------------------------

    def _on_autofill_clicked(self) -> None:
        """Autofill asset name, domicile, and category using OpenFIGI enrichment."""
        isin = self._input_isin.text().strip().upper()
        if not isin or len(isin) != 12:
            QMessageBox.warning(self, "Invalid ISIN", "Please enter a valid 12-character ISIN code.")
            return

        figi_res = self._figi_mapper.map_isin(isin)
        ticker, name = figi_res.ticker, figi_res.name
        if name:
            self._input_name.setText(name)
            detected_category, detected_domicile = _detect_asset_traits_from_metadata(isin, name)

            cat_idx = self._combo_category.findText(detected_category)
            if cat_idx >= 0:
                self._combo_category.setCurrentIndex(cat_idx)

            if detected_domicile:
                self._combo_domicile.setEditText(detected_domicile)

            self._auto_infer_tax_regime()
            QMessageBox.information(
                self,
                "OpenFIGI Result",
                f"Found ticker '{ticker}' and name '{name}'. Detected category: '{detected_category}'.",
            )
        else:
            QMessageBox.warning(
                self,
                "OpenFIGI Not Found",
                f"Could not find OpenFIGI data for ISIN {isin}. You can enter details manually.",
            )

    def _on_save_clicked(self) -> None:
        """Validate, construct domain model, and persist asset classification to database."""
        isin = self._input_isin.text().strip().upper()
        asset_name = self._input_name.text().strip() or None
        domicile = self._combo_domicile.currentText().strip().upper()
        if not domicile:
            domicile = isin[:2]
        if len(domicile) != 2:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Domicile country must be a valid 2-letter ISO country code (e.g. IE, LU, US).",
            )
            return

        notes = self._input_notes.text().strip() or None
        category_val = self._combo_category.currentText()
        regime_val = self._combo_regime.currentText()

        if not isin or len(isin) != 12:
            QMessageBox.warning(self, "Validation Error", "ISIN code must be exactly 12 characters.")
            return

        try:
            tax_regime = parse_irish_tax_regime(regime_val)
        except ValueError:
            QMessageBox.warning(self, "Validation Error", f"Invalid tax regime '{regime_val}'.")
            return

        # Check existing record in DB
        existing_db = self._db.get_asset_tax_classification(isin)

        # If Adding new item and ISIN already exists, confirm overwrite
        if self._selected_item is None and existing_db is not None:
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Classification?",
                f"A classification for ISIN {isin} already exists. Do you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # If Editing item and ISIN was changed from previous
        if self._selected_item is not None and self._selected_item.isin != isin:
            reply = QMessageBox.question(
                self,
                "ISIN Changed",
                f"You modified the ISIN from {self._selected_item.isin} to {isin}. "
                "Do you want to save this as a new asset classification record?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        domain_record = AssetTaxClassificationDomain(
            isin=isin,
            asset_name=asset_name,
            tax_regime=tax_regime,
            domicile_country=domicile,
            is_ucits=category_val == CATEGORY_UCITS_ETF or tax_regime == IrishTaxRegime.EXIT_TAX,
            is_etc=category_val == CATEGORY_ETC or tax_regime == IrishTaxRegime.ETC_COMMODITY,
            is_offshore_distributing=(
                category_val == CATEGORY_OFFSHORE_DIST or tax_regime == IrishTaxRegime.OFFSHORE_DISTRIBUTING
            ),
            classification_source="manual",
            notes=notes,
        )

        db_entity = AssetTaxClassification.from_domain(domain_record)

        try:
            self._db.upsert_asset_tax_classification(db_entity)
            QMessageBox.information(self, "Success", f"Saved asset classification for {isin}.")
            self._set_add_mode()
            self.load_classifications()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save asset classification: {exc}")
