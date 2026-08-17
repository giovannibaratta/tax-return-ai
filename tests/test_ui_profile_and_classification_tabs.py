"""Tests for TaxpayerProfileTab and AssetClassificationTab UI components."""

from decimal import Decimal

import pytest
from PySide6.QtWidgets import QApplication
from sqlmodel import SQLModel

from backend.db_manager import DatabaseManager, MemoryDb
from src.jurisdiction.ireland.cgt_models import (
    AssetTaxClassification,
    AssetTaxClassificationDomain,
    IrishTaxRegime,
    ResidencyType,
    TaxpayerProfile,
    infer_residency_type,
    parse_irish_tax_regime,
)
from src.ui.classification_tab import (
    CATEGORY_ETC,
    AssetClassificationTab,
)
from src.ui.config import UIConfig
from src.ui.profile_tab import TaxpayerProfileTab


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Share a single QApplication instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # pyright: ignore[reportReturnType]


@pytest.fixture
def test_db():
    db = DatabaseManager(MemoryDb())
    SQLModel.metadata.create_all(db.engine)
    yield db
    db.close()


def test_taxpayer_profile_tab_ui(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: Seed DB with profile
    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="IT",
        residency_type=ResidencyType.RESIDENT_NON_DOMICILED.value,
        marginal_tax_rate=Decimal("0.40"),
        notes="Test note",
    )
    test_db.upsert_taxpayer_profile(profile)

    # When: Tab created
    tab = TaxpayerProfileTab(db=test_db)

    # Then: Table has 1 row with loaded profile
    assert tab._table.rowCount() == 1
    item_year = tab._table.item(0, 0)
    item_res = tab._table.item(0, 1)
    item_dom = tab._table.item(0, 2)
    item_is_dom = tab._table.item(0, 3)
    assert item_year is not None and item_year.text() == "2025"
    assert item_res is not None and item_res.text() == "IE"
    assert item_dom is not None and item_dom.text() == "IT"
    assert item_is_dom is not None and item_is_dom.text() == "No"  # is_domiciled_in_ireland property


def test_asset_classification_tab_ui(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: Seed DB with classification
    classification = AssetTaxClassification(
        isin="IE00BFWXDV39",
        asset_name="Vanguard S&P 500 UCITS ETF",
        tax_regime=IrishTaxRegime.EXIT_TAX.value,
        domicile_country="IE",
        is_ucits=True,
        classification_source="manual",
    )
    test_db.upsert_asset_tax_classification(classification)

    # When: Tab created with UIConfig
    config = UIConfig(db=test_db)
    tab = AssetClassificationTab(config=config)

    # Then: Table has 1 row with loaded classification
    assert tab._table.rowCount() == 1
    item_isin = tab._table.item(0, 0)
    item_regime = tab._table.item(0, 2)
    item_ucits = tab._table.item(0, 4)
    assert item_isin is not None and item_isin.text() == "IE00BFWXDV39"
    assert item_regime is not None and item_regime.text() == "exit_tax"
    assert item_ucits is not None and item_ucits.text() == "Yes"


def test_asset_classification_domain_model_conversion() -> None:
    # Given: A domain classification model
    domain = AssetTaxClassificationDomain(
        isin="IE00BFWXDV39",
        asset_name="Vanguard S&P 500 UCITS ETF",
        tax_regime=IrishTaxRegime.EXIT_TAX,
        domicile_country="IE",
        is_ucits=True,
        is_etc=False,
        is_offshore_distributing=False,
        classification_source="manual",
        notes="Test note",
    )

    # When: Converted to DB model and back to domain
    db_entity = AssetTaxClassification.from_domain(domain)
    converted_domain = db_entity.to_domain()

    # Then: Attributes match precisely
    assert db_entity.isin == "IE00BFWXDV39"
    assert db_entity.tax_regime == "exit_tax"
    assert db_entity.is_ucits is True
    assert converted_domain.isin == domain.isin
    assert converted_domain.tax_regime == domain.tax_regime
    assert converted_domain.is_ucits is True


def test_asset_classification_tab_modes_and_dirty_tracking(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: Tab created with clean state and 1 seed item
    classification = AssetTaxClassification(
        isin="IE00B4L5Y983",
        asset_name="iShares Core MSCI World UCITS ETF",
        tax_regime=IrishTaxRegime.EXIT_TAX.value,
        domicile_country="IE",
        is_ucits=True,
        classification_source="manual",
    )
    test_db.upsert_asset_tax_classification(classification)
    config = UIConfig(db=test_db)
    tab = AssetClassificationTab(config=config)

    # When: Fresh tab in Add Mode
    # Then: Not dirty, mode is Add
    assert not tab._is_form_dirty()
    assert "Adding New Asset" in tab._lbl_mode.text()

    # When: Typing into form in Add Mode
    tab._input_isin.setText("US0378331005")
    # Then: Marked dirty
    assert tab._is_form_dirty()

    # When: Deselect / Add New clicked
    tab._set_add_mode()
    assert not tab._is_form_dirty()

    # When: Selecting existing row
    tab._table.selectRow(0)
    # Then: Enters Edit Mode, loads record, not dirty initially
    assert tab._selected_item is not None
    assert tab._selected_item.isin == "IE00B4L5Y983"
    assert "Editing ISIN [IE00B4L5Y983]" in tab._lbl_mode.text()
    assert not tab._is_form_dirty()

    # When: Changing notes in Edit Mode
    tab._input_notes.setText("Modified note")
    # Then: Marked dirty
    assert tab._is_form_dirty()


def test_asset_classification_tab_regime_inference_and_override(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: Tab created
    config = UIConfig(db=test_db)
    tab = AssetClassificationTab(config=config)

    # When: Selecting ETC category
    idx = tab._combo_category.findText(CATEGORY_ETC)
    tab._combo_category.setCurrentIndex(idx)

    # Then: Regime inferred as etc_commodity and combo is disabled
    assert tab._combo_regime.currentText() == IrishTaxRegime.ETC_COMMODITY.value
    assert not tab._combo_regime.isEnabled()
    assert "etc_commodity" in tab._lbl_inferred_regime.text()

    # When: Checking manual override
    tab._chk_override_regime.setChecked(True)
    # Then: Combo is enabled and user can select another regime
    assert tab._combo_regime.isEnabled()
    tab._combo_regime.setCurrentIndex(tab._combo_regime.findText(IrishTaxRegime.CGT_STANDARD.value))
    assert tab._combo_regime.currentText() == IrishTaxRegime.CGT_STANDARD.value

    # When: Unchecking override
    tab._chk_override_regime.setChecked(False)
    # Then: Reverts to inferred regime and is disabled
    assert tab._combo_regime.currentText() == IrishTaxRegime.ETC_COMMODITY.value
    assert not tab._combo_regime.isEnabled()


def test_infer_residency_type_logic(qapp: QApplication, test_db: DatabaseManager) -> None:
    # IE residence + IE domicile -> RESIDENT_DOMICILED
    assert infer_residency_type("IE", "IE") == ResidencyType.RESIDENT_DOMICILED

    # IE residence + IT domicile -> RESIDENT_NON_DOMICILED
    assert infer_residency_type("IE", "IT") == ResidencyType.RESIDENT_NON_DOMICILED

    # IT residence + IT domicile -> NON_RESIDENT
    assert infer_residency_type("IT", "IT") == ResidencyType.NON_RESIDENT

    # Auto-inference in tab UI
    tab = TaxpayerProfileTab(db=test_db)
    tab._combo_residence.setEditText("IE")
    tab._combo_domicile.setEditText("IT")
    assert tab._combo_residency_type.currentText() == ResidencyType.RESIDENT_NON_DOMICILED.value


def test_parse_irish_tax_regime_case_and_name_support() -> None:
    assert parse_irish_tax_regime("exit_tax") == IrishTaxRegime.EXIT_TAX
    assert parse_irish_tax_regime("EXIT_TAX") == IrishTaxRegime.EXIT_TAX
    assert parse_irish_tax_regime("cgt_standard") == IrishTaxRegime.CGT_STANDARD
    assert parse_irish_tax_regime("CGT_STANDARD") == IrishTaxRegime.CGT_STANDARD
    assert parse_irish_tax_regime(IrishTaxRegime.ETC_COMMODITY) == IrishTaxRegime.ETC_COMMODITY

    with pytest.raises(ValueError, match="is not a valid IrishTaxRegime"):
        parse_irish_tax_regime("invalid_regime")


def test_db_loading_with_uppercase_regime_name(qapp: QApplication, test_db: DatabaseManager) -> None:
    # Given: DB populated with uppercase 'EXIT_TAX' as tax_regime string
    classification = AssetTaxClassification(
        isin="IE00BFWXDV39",
        asset_name="Vanguard S&P 500 UCITS ETF",
        tax_regime="EXIT_TAX",
        domicile_country="IE",
        is_ucits=True,
    )
    test_db.upsert_asset_tax_classification(classification)

    # When: Tab created with UIConfig
    config = UIConfig(db=test_db)
    tab = AssetClassificationTab(config=config)

    # Then: Loads successfully and maps to domain model
    assert len(tab._classifications) == 1
    assert tab._classifications[0].tax_regime == IrishTaxRegime.EXIT_TAX
    assert tab._classifications[0].domicile_country == "IE"
