"""Unit tests for Irish Capital Gains Tax computation engine.

Uses an in-memory SQLite DatabaseManager instance to ensure complete
isolation from production data.
"""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlmodel import SQLModel

from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import FinancialRecord
from backend.domain_models import AssetIdentity, AssetType, TradeRecord, TransactionAction, VerificationStatus
from src.jurisdiction.ireland.cgt_engine import (
    apply_section581_quarantine,
    determine_irish_taxability,
    get_tax_rate,
    match_lots_fifo,
)
from src.jurisdiction.ireland.cgt_models import (
    AssetTaxClassification,
    IrishTaxRegime,
    LotMatch,
    RemittanceEvent,
    ResidencyType,
    SimulatedDisposalInput,
    StrictDisposalInput,
    TaxpayerProfile,
    infer_tax_regime,
)
from src.jurisdiction.ireland.orchestrator import (
    classify_tax_regime,
    compute_deemed_disposals,
    compute_disposal,
    detect_section581_repurchase,
    get_remitted_amount,
    load_buy_lots,
)
from tests.utils import insert_financial_record


def test_infer_tax_regime_explicit_and_ambiguous() -> None:
    """Test explicit regime inference and ValueError on ambiguous traits."""
    # Given: explicit UCITS trait
    assert (
        infer_tax_regime(
            is_ucits=True,
            is_etc=False,
            is_offshore_distributing=False,
            is_direct_equity_or_crypto=False,
        )
        == IrishTaxRegime.EXIT_TAX
    )

    # Given: explicit direct equity trait
    assert (
        infer_tax_regime(
            is_ucits=False,
            is_etc=False,
            is_offshore_distributing=False,
            is_direct_equity_or_crypto=True,
        )
        == IrishTaxRegime.CGT_STANDARD
    )

    # Given: ambiguous unconfirmed traits (all False)
    with pytest.raises(ValueError, match="unconfirmed or ambiguous"):
        infer_tax_regime(
            is_ucits=False,
            is_etc=False,
            is_offshore_distributing=False,
            is_direct_equity_or_crypto=False,
        )


@pytest.fixture
def test_db():
    db = DatabaseManager(MemoryDb())
    # Ensure tables are created
    SQLModel.metadata.create_all(db.engine)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# 1. Tax Regime Classification Tests
# ---------------------------------------------------------------------------


def test_classify_tax_regime_exit_tax(test_db: DatabaseManager):
    # Given: ISIN classified as exit_tax
    cls = AssetTaxClassification(
        isin="IE00BFWXDV39",
        asset_name="Vanguard UCITS ETF",
        tax_regime=IrishTaxRegime.EXIT_TAX,
        domicile_country="IE",
        is_ucits=True,
    )
    test_db.upsert_asset_tax_classification(cls)

    # When: classify_tax_regime called
    res = classify_tax_regime(test_db, "IE00BFWXDV39")

    # Then: returns classification record
    assert res.tax_regime == IrishTaxRegime.EXIT_TAX.value
    assert res.is_ucits is True


def test_classify_tax_regime_cgt_standard(test_db: DatabaseManager):
    # Given: ISIN classified as cgt_standard
    cls = AssetTaxClassification(
        isin="US0378331005",
        asset_name="Apple Inc",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="US",
        is_ucits=False,
    )
    test_db.upsert_asset_tax_classification(cls)

    # When: classify_tax_regime called
    res = classify_tax_regime(test_db, "US0378331005")

    # Then: returns cgt_standard
    assert res.tax_regime == IrishTaxRegime.CGT_STANDARD.value


def test_classify_tax_regime_unknown_isin_hard_fails(test_db: DatabaseManager):
    # Given: ISIN not in database
    # When/Then: raises ValueError
    with pytest.raises(ValueError, match="not found in asset_tax_classification table"):
        classify_tax_regime(test_db, "UNKNOWN_ISIN")


@pytest.mark.parametrize(
    "regime, event_date, tax_year, expected_rate",
    [
        (IrishTaxRegime.EXIT_TAX, date(2025, 6, 1), 2025, Decimal("0.41")),
        (IrishTaxRegime.EXIT_TAX, date(2026, 1, 15), 2026, Decimal("0.38")),
        (IrishTaxRegime.CGT_STANDARD, date(2025, 6, 1), 2025, Decimal("0.33")),
        (IrishTaxRegime.OFFSHORE_NON_DISTRIBUTING, date(2025, 6, 1), 2025, Decimal("0.40")),
        (IrishTaxRegime.ETC_COMMODITY, date(2025, 6, 1), 2025, Decimal("0.33")),
        (IrishTaxRegime.OFFSHORE_DISTRIBUTING, date(2025, 6, 1), 2025, Decimal("0.40")),
    ],
)
def test_get_tax_rate_resolution(
    regime: IrishTaxRegime,
    event_date: date,
    tax_year: int,
    expected_rate: Decimal,
) -> None:
    # Given: A resident domiciled profile with 40% marginal rate
    profile = TaxpayerProfile(
        tax_year=tax_year,
        fiscal_residence_country="IE",
        domicile_country="IE",
        residency_type=ResidencyType.RESIDENT_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )

    # When: Resolving statutory tax rate
    rate = get_tax_rate(regime, event_date, profile)

    # Then: Matches expected statutory rate
    assert rate == expected_rate


# ---------------------------------------------------------------------------
# 2. Taxpayer Profile & Taxability Tests
# ---------------------------------------------------------------------------


def test_taxability_resident_domiciled():
    # Given: Resident domiciled profile (worldwide taxability)
    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="IE",
        residency_type=ResidencyType.RESIDENT_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )
    cls = AssetTaxClassification(
        isin="US0378331005",
        asset_name="Apple Inc",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="US",
    )

    taxable, remittance = determine_irish_taxability(
        profile, cls, is_remitted=False, is_irish_specified_asset=False
    )

    assert taxable is True
    assert remittance is False


def test_taxability_resident_non_domiciled_foreign_asset_remitted():
    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="IT",
        residency_type=ResidencyType.RESIDENT_NON_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )
    cls = AssetTaxClassification(
        isin="US0378331005",
        asset_name="Apple Inc",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="US",
    )

    # When: Checking taxability when proceeds are remitted to Ireland
    taxable, remittance_basis_applies = determine_irish_taxability(
        profile, cls, is_remitted=True, is_irish_specified_asset=False
    )

    # Then: Foreign asset remitted to Ireland is taxable under the remittance basis
    assert taxable is True
    assert remittance_basis_applies is True


def test_taxability_resident_non_domiciled_foreign_asset_not_remitted() -> None:
    # Given: Resident non-domiciled profile and foreign asset (US domicile)
    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="IT",
        residency_type=ResidencyType.RESIDENT_NON_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )
    cls = AssetTaxClassification(
        isin="US0378331005",
        asset_name="Apple Inc",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="US",
    )

    # When: Checking taxability when proceeds are not remitted
    taxable, remittance_basis_applies = determine_irish_taxability(
        profile, cls, is_remitted=False, is_irish_specified_asset=False
    )

    # Then: Foreign asset NOT remitted is not taxable in Ireland, but remittance basis rules applied
    assert taxable is False
    assert remittance_basis_applies is True


def test_taxability_resident_non_domiciled_irish_situated_asset():
    # Given: A resident non-domiciled profile and an Irish-situated asset (domicile_country='IE')
    # Irish-situated assets are always taxable for Irish residents regardless of remittance basis
    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="US",
        residency_type=ResidencyType.RESIDENT_NON_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )
    cls = AssetTaxClassification(
        isin="IE00B4ND3602",
        asset_name="Irish Stock",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="IE",
    )

    # When: Taxability is checked without remittance (is_remitted=False)
    taxable, remittance_applies = determine_irish_taxability(
        profile, cls, is_remitted=False, is_irish_specified_asset=False
    )

    # Then: Irish-situated assets are fully taxable regardless of remittance
    assert taxable is True
    assert remittance_applies is False


def test_taxability_non_resident_ordinary_shares() -> None:
    # Given: Non-resident taxpayer disposing of ordinary foreign/quoted shares
    profile = TaxpayerProfile(
        tax_year=2024,
        fiscal_residence_country="IT",
        domicile_country="IT",
        residency_type=ResidencyType.NON_RESIDENT,
        marginal_tax_rate=Decimal("0.40"),
    )
    cls = AssetTaxClassification(
        isin="US0378331005",
        asset_name="Apple Inc",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="US",
    )

    # When: Checking taxability
    taxable, remittance_applies = determine_irish_taxability(
        profile, cls, is_remitted=True, is_irish_specified_asset=False
    )

    # Then: Non-residents are not taxable in Ireland on ordinary foreign/quoted shares (Section 29 TCA 1997)
    assert taxable is False
    assert remittance_applies is False


def test_taxability_non_resident_irish_specified_asset() -> None:
    # Given: Non-resident taxpayer disposing of an Irish specified asset (e.g. unquoted shares deriving >50% value from Irish land)
    profile = TaxpayerProfile(
        tax_year=2024,
        fiscal_residence_country="IT",
        domicile_country="IT",
        residency_type=ResidencyType.NON_RESIDENT,
        marginal_tax_rate=Decimal("0.40"),
    )
    cls = AssetTaxClassification(
        isin="IE00SPECIFIED1",
        asset_name="Irish Land Holdings Ltd",
        tax_regime=IrishTaxRegime.CGT_STANDARD,
        domicile_country="IE",
    )

    # When: Checking taxability with is_irish_specified_asset=True
    taxable, remittance_applies = determine_irish_taxability(
        profile, cls, is_remitted=False, is_irish_specified_asset=True
    )

    # Then: Non-residents are chargeable to Irish CGT on Irish specified assets per Section 29(3) TCA 1997
    assert taxable is True
    assert remittance_applies is False


# ---------------------------------------------------------------------------
# 3. FIFO Lot Matching & Section 580 Tests
# ---------------------------------------------------------------------------


def test_single_lot_sell_gain(test_db: DatabaseManager):
    # Given: 10 shares AAPL bought at €100
    buy = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha1",
        event_timestamp=datetime(2025, 1, 10, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("1000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1000"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)

    # When: load_buy_lots called
    lots = load_buy_lots(test_db, "US0378331005")

    # Then: returns 1 buy lot
    assert len(lots) == 1
    assert lots[0].quantity == Decimal("10")
    assert lots[0].unit_price == Decimal("100")


def test_cross_jurisdiction_fifo(test_db: DatabaseManager):
    # Given: Lot 1 bought via Italian broker in 2023, Lot 2 bought via Irish broker in 2025
    buy1 = FinancialRecord(
        provider="directa",
        source_file_sha="sha1",
        event_timestamp=datetime(2023, 5, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("5"),
        unit_price=Decimal("80"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("400"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("400"),
        tax_year=2023,
        account_country="italy",
        verification_status="approved",
    )
    buy2 = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha2",
        event_timestamp=datetime(2025, 2, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("5"),
        unit_price=Decimal("120"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("600"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("600"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy1)
    insert_financial_record(test_db, buy2)

    # When: load_buy_lots called for US0378331005
    lots = load_buy_lots(test_db, "US0378331005")

    # Then: both lots returned in FIFO order (Italy lot first)
    assert len(lots) == 2
    assert lots[0].account_country == "italy"
    assert lots[0].unit_price == Decimal("80")
    assert lots[1].account_country == "ireland"
    assert lots[1].unit_price == Decimal("120")


# ---------------------------------------------------------------------------
# 4. Section 581 Bed-and-Breakfasting Tests
# ---------------------------------------------------------------------------


def test_section581_loss_quarantined(test_db: DatabaseManager):
    # Given: Sell on 2025-03-01, Buy back on 2025-03-15 (within 28 days)
    disposal_dt = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    repurchase = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha3",
        event_timestamp=datetime(2025, 3, 15, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("90"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("900"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("900"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, repurchase)

    # When: detect_section581_repurchase called
    reps = detect_section581_repurchase(test_db, "US0378331005", disposal_dt)

    # Then: repurchase detected
    assert len(reps) == 1
    assert reps[0].id == repurchase.id


@pytest.mark.parametrize(
    "repurchase_date, expected_detected",
    [
        (datetime(2025, 3, 28, 10, 0, tzinfo=timezone.utc), True),  # Day +27 (inside window)
        (datetime(2025, 3, 29, 10, 0, tzinfo=timezone.utc), True),  # Day +28 (exact boundary)
        (datetime(2025, 3, 30, 10, 0, tzinfo=timezone.utc), False),  # Day +29 (outside window)
        (datetime(2025, 4, 10, 10, 0, tzinfo=timezone.utc), False),  # Day +40 (well outside window)
    ],
)
def test_detect_section581_repurchase_window_boundaries(
    test_db: DatabaseManager,
    repurchase_date: datetime,
    expected_detected: bool,
) -> None:
    # Given: Disposal on 2025-03-01 and repurchase on parameterized date
    disposal_dt = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    repurchase = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha_rep",
        event_timestamp=repurchase_date,
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("90"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("900"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("900"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, repurchase)

    # When: detect_section581_repurchase is called
    reps = detect_section581_repurchase(test_db, "US0378331005", disposal_dt)

    # Then: Detected only when within the 28-day window
    assert (len(reps) > 0) is expected_detected


# ---------------------------------------------------------------------------
# 5. compute_disposal Orchestrator Tests
# ---------------------------------------------------------------------------


def test_compute_disposal_simulated_cgt_gain(test_db: DatabaseManager):
    # Given: DB setup with taxpayer profile, classification, and 1 buy lot
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="US0378331005",
            asset_name="Apple Inc",
            tax_regime=IrishTaxRegime.CGT_STANDARD,
            domicile_country="US",
            is_ucits=False,
        )
    )
    buy = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha1",
        event_timestamp=datetime(2025, 1, 10, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("10"),
        total_amount=Decimal("1010"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1010"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)

    # When: Simulated sale of 10 AAPL at €150 with €5 fee on 2025-06-15
    sim_input = SimulatedDisposalInput(
        isin="US0378331005",
        quantity=Decimal("10"),
        disposal_date=date(2025, 6, 15),
        estimated_unit_price_eur=Decimal("150"),
        estimated_fees_eur=Decimal("5"),
    )
    res = compute_disposal(test_db, sim_input)

    # Then: Total proceeds = 1500 - 5 = 1495. Cost basis = 1000 + 10 = 1010. Realized gain = 485.
    assert res.isin == "US0378331005"
    assert res.tax_regime == IrishTaxRegime.CGT_STANDARD
    assert res.total_quantity == Decimal("10")
    assert res.total_proceeds_eur == Decimal("1495.00")
    assert res.total_cost_basis_eur == Decimal("1010.00")
    assert res.gross_gain_loss_eur == Decimal("485.00")
    assert res.applicable_tax_rate == Decimal("0.33")
    assert res.annual_exemption_applicable is True
    assert res.is_simulation is True


def test_compute_disposal_strict_mode(test_db: DatabaseManager):
    # Given: DB setup with profile, classification, buy record, and sell record
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="US0378331005",
            asset_name="Apple Inc",
            tax_regime=IrishTaxRegime.CGT_STANDARD,
            domicile_country="US",
            is_ucits=False,
        )
    )
    buy = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha1",
        event_timestamp=datetime(2025, 1, 10, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="buy",
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("1000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1000"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    sell = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha2",
        event_timestamp=datetime(2025, 7, 20, 10, 0, tzinfo=timezone.utc),
        asset_type="stock",
        symbol="AAPL",
        isin="US0378331005",
        action="sell",
        quantity=Decimal("10"),
        unit_price=Decimal("160"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("1600"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1600"),
        tax_year=2025,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)
    inserted_sell = insert_financial_record(test_db, sell)
    assert inserted_sell.id is not None

    # When: StrictDisposalInput called
    res = compute_disposal(test_db, StrictDisposalInput(record_id=inserted_sell.id))

    # Then: gain calculated from database records
    assert res.disposal_record_id == inserted_sell.id
    assert res.gross_gain_loss_eur == Decimal("600.00")
    assert res.is_simulation is False


# ---------------------------------------------------------------------------
# 6. Deemed Disposal Tests
# ---------------------------------------------------------------------------


def test_deemed_disposal_scanner(test_db: DatabaseManager):
    # Given: ETF lot acquired on 2017-01-01 (over 8 years old as of 2025-06-01)
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="IE00BFWXDV39",
            asset_name="Vanguard UCITS ETF",
            tax_regime=IrishTaxRegime.EXIT_TAX,
            domicile_country="IE",
            is_ucits=True,
        )
    )
    buy = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha1",
        event_timestamp=datetime(2017, 1, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="buy",
        quantity=Decimal("100"),
        unit_price=Decimal("50"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("5000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("5000"),
        tax_year=2017,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)

    # Orchestrator compute_deemed_disposals handles DB I/O and delegates to pure cgt_engine.compute_deemed_disposals.
    # When: compute_deemed_disposals called as of 2025-06-01 with current price €80
    res = compute_deemed_disposals(
        test_db,
        evaluation_date=date(2025, 6, 1),
        market_prices={"IE00BFWXDV39": Decimal("80")},
    )

    # Then: 1 event triggered. Deemed gain = 100 * (80 - 50) = 3000. Exit tax at 41% (2025 trigger) = 1230.
    assert len(res.events) == 1
    event = res.events[0]
    assert event.isin == "IE00BFWXDV39"
    assert event.deemed_gain_eur == Decimal("3000.00")
    assert event.exit_tax_due_eur == Decimal("1230.00")
    assert event.stepped_up_cost_per_unit_eur == Decimal("80")
    assert res.total_exit_tax_due_eur == Decimal("1230.00")


def test_determine_irish_taxability_exit_tax_bypasses_remittance():
    # Given: Non-domiciled resident profile and Exit Tax classification
    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="IT",
        residency_type=ResidencyType.RESIDENT_NON_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )
    classification = AssetTaxClassification(
        isin="IE00BFWXDV39",
        asset_name="Vanguard UCITS ETF",
        tax_regime=IrishTaxRegime.EXIT_TAX,
        domicile_country="IE",
        is_ucits=True,
    )

    # When: determine_irish_taxability called (even if not remitted)
    taxable, remittance_applies = determine_irish_taxability(
        profile, classification, is_remitted=False, is_irish_specified_asset=False
    )

    # Then: Exit tax asset is taxable in Ireland immediately, remittance basis does NOT apply
    assert taxable is True
    assert remittance_applies is False


def test_section581_preceding_28_days_matching_priority(test_db: DatabaseManager):
    # Given: Old buy in 2020 (10 units @ €100), recent buy on 2025-02-15 (5 units @ €150)
    # Disposal on 2025-03-01 of 5 units @ €200.
    # Per S. 581(1), recent buy (within 28 days preceding) must be matched FIRST.
    disposal_dt = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    old_buy = TradeRecord(
        id=1,
        provider="interactive_brokers",
        ingestion_timestamp=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
        source_file_sha="sha1",
        event_timestamp=datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("1000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1000"),
        tax_year=2020,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )
    recent_buy = TradeRecord(
        id=2,
        provider="interactive_brokers",
        ingestion_timestamp=datetime(2025, 2, 15, 0, 0, tzinfo=timezone.utc),
        source_file_sha="sha2",
        event_timestamp=datetime(2025, 2, 15, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("5"),
        unit_price=Decimal("150"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("750"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("750"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )

    # When: match_lots_fifo called for sell of 5 units
    matches = match_lots_fifo(
        buy_lots=[old_buy, recent_buy],
        sell_quantity=Decimal("5"),
        sell_unit_price_eur=Decimal("200"),
        sell_fees_eur=Decimal("0"),
        sell_date=disposal_dt,
    )

    # Then: matched against recent_buy (ID 2) @ €150 cost basis, NOT old_buy (ID 1)
    assert len(matches) == 1
    assert matches[0].source_record_id == 2
    assert matches[0].cost_basis_eur == Decimal("750")
    assert matches[0].gain_loss_eur == Decimal("250")


def test_section581_proportional_loss_quarantine():
    # Given: Sold 10 units at a loss of €500 total (€50 loss per unit).
    # Repurchased 4 units within 28 days.
    # Per S. 581(3), loss restricted on 4 units (€200), unrestricted on 6 units (€300).
    lot_matches = [
        LotMatch(
            source_record_id=1,
            source_account_country="ireland",
            buy_date=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            matched_quantity=Decimal("10"),
            cost_basis_eur=Decimal("1000"),
            proceeds_eur=Decimal("500"),
            gain_loss_eur=Decimal("-500"),
        )
    ]
    repurchase_record = TradeRecord(
        id=99,
        provider="interactive_brokers",
        ingestion_timestamp=datetime(2025, 3, 10, 0, 0, tzinfo=timezone.utc),
        source_file_sha="sha3",
        event_timestamp=datetime(2025, 3, 10, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("4"),
        unit_price=Decimal("50"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("200"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("200"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )

    # When: apply_section581_quarantine called
    result = apply_section581_quarantine(lot_matches, [repurchase_record])

    # Then: split into 2 match chunks: 4 restricted units (€200 loss) and 6 unrestricted units (€300 loss)
    assert len(result) == 2
    restricted = next(m for m in result if m.is_section581_restricted)
    unrestricted = next(m for m in result if not m.is_section581_restricted)

    assert restricted.matched_quantity == Decimal("4")
    assert restricted.gain_loss_eur == Decimal("-200")
    assert unrestricted.matched_quantity == Decimal("6")
    assert unrestricted.gain_loss_eur == Decimal("-300")


def test_section581_preceding_28_days_earlier_acquisition_priority(test_db: DatabaseManager) -> None:
    # Given: Buy 1 on Day -20 (5 units @ €100), Buy 2 on Day -5 (5 units @ €120). Sell 5 units on Day 0 @ €150.
    # Per Section 581(2) TCA 1997, shares acquired within the 4-week window are identified with shares
    # acquired at an EARLIER date first (Buy 1 @ €100 matched first).
    disposal_dt = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    buy_day_20 = TradeRecord(
        id=1,
        provider="interactive_brokers",
        ingestion_timestamp=datetime(2025, 2, 9, 0, 0, tzinfo=timezone.utc),
        source_file_sha="sha1",
        event_timestamp=datetime(2025, 2, 9, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("5"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("500"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("500"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )
    buy_day_5 = TradeRecord(
        id=2,
        provider="interactive_brokers",
        ingestion_timestamp=datetime(2025, 2, 24, 0, 0, tzinfo=timezone.utc),
        source_file_sha="sha2",
        event_timestamp=datetime(2025, 2, 24, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("5"),
        unit_price=Decimal("120"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("600"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("600"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )

    # When: match_lots_fifo called for sell of 5 units
    matches = match_lots_fifo(
        buy_lots=[buy_day_20, buy_day_5],
        sell_quantity=Decimal("5"),
        sell_unit_price_eur=Decimal("150"),
        sell_fees_eur=Decimal("0"),
        sell_date=disposal_dt,
    )

    # Then: matched against buy_day_20 (ID 1 @ €100) per S.581(2), realized gain = 5 * (150 - 100) = 250
    assert len(matches) == 1
    assert matches[0].source_record_id == 1
    assert matches[0].cost_basis_eur == Decimal("500")
    assert matches[0].gain_loss_eur == Decimal("250")


def test_section581_preceding_window_boundary_matching() -> None:
    # Given: Disposal on 2025-03-01 of 5 units @ €200.
    # Buy 1 on Day -29 (2025-01-31, 5 units @ €100) -> outside 28 days
    # Buy 2 on Day -28 (2025-02-01, 5 units @ €140) -> inside 28-day window boundary
    # Buy 3 on Day -27 (2025-02-02, 5 units @ €160) -> inside 28-day window
    disposal_dt = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    buy_day_29 = TradeRecord(
        id=1,
        provider="ibkr",
        ingestion_timestamp=datetime(2025, 1, 31, 0, 0, tzinfo=timezone.utc),
        source_file_sha="s1",
        event_timestamp=datetime(2025, 1, 31, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("5"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("500"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("500"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )
    buy_day_28 = TradeRecord(
        id=2,
        provider="ibkr",
        ingestion_timestamp=datetime(2025, 2, 1, 0, 0, tzinfo=timezone.utc),
        source_file_sha="s2",
        event_timestamp=datetime(2025, 2, 1, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("5"),
        unit_price=Decimal("140"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("700"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("700"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )
    buy_day_27 = TradeRecord(
        id=3,
        provider="ibkr",
        ingestion_timestamp=datetime(2025, 2, 2, 0, 0, tzinfo=timezone.utc),
        source_file_sha="s3",
        event_timestamp=datetime(2025, 2, 2, 10, 0, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("5"),
        unit_price=Decimal("160"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("800"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("800"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )

    # When: Matching 5 units sold
    matches = match_lots_fifo(
        buy_lots=[buy_day_29, buy_day_28, buy_day_27],
        sell_quantity=Decimal("5"),
        sell_unit_price_eur=Decimal("200"),
        sell_fees_eur=Decimal("0"),
        sell_date=disposal_dt,
    )

    # Then: Matched against Buy Day -28 (ID 2) per S.581(1) and S.581(2)
    assert len(matches) == 1
    assert matches[0].source_record_id == 2
    assert matches[0].cost_basis_eur == Decimal("700")
    assert matches[0].gain_loss_eur == Decimal("300")


def test_deemed_disposal_8_year_boundaries(test_db: DatabaseManager) -> None:
    # Given: 100 units acquired on 2017-01-15 @ €50
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="IE00BFWXDV39",
            asset_name="Vanguard UCITS ETF",
            tax_regime=IrishTaxRegime.EXIT_TAX,
            domicile_country="IE",
            is_ucits=True,
        )
    )
    buy = FinancialRecord(
        provider="ibkr",
        source_file_sha="sha1",
        event_timestamp=datetime(2017, 1, 15, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="buy",
        quantity=Decimal("100"),
        unit_price=Decimal("50"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("5000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("5000"),
        tax_year=2017,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)

    # Case 1: Evaluation at 7 years 364 days (2025-01-14) -> No event
    res_before = compute_deemed_disposals(
        test_db,
        evaluation_date=date(2025, 1, 14),
        market_prices={"IE00BFWXDV39": Decimal("80")},
    )
    assert len(res_before.events) == 0

    # Case 2: Evaluation at exactly 8 years (2025-01-15) -> 1 event triggered
    res_exact = compute_deemed_disposals(
        test_db,
        evaluation_date=date(2025, 1, 15),
        market_prices={"IE00BFWXDV39": Decimal("80")},
    )
    assert len(res_exact.events) == 1
    assert res_exact.events[0].deemed_gain_eur == Decimal("3000.00")
    assert res_exact.events[0].exit_tax_due_eur == Decimal("1230.00")
    assert res_exact.events[0].stepped_up_cost_per_unit_eur == Decimal("80")


def test_deemed_disposal_subsequent_16_year_cycle(test_db: DatabaseManager) -> None:
    # Given: 100 units acquired on 2017-01-15 @ €50
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2033,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="IE00BFWXDV39",
            asset_name="Vanguard UCITS ETF",
            tax_regime=IrishTaxRegime.EXIT_TAX,
            domicile_country="IE",
            is_ucits=True,
        )
    )
    buy = FinancialRecord(
        provider="ibkr",
        source_file_sha="sha1",
        event_timestamp=datetime(2017, 1, 15, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="buy",
        quantity=Decimal("100"),
        unit_price=Decimal("50"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("5000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("5000"),
        tax_year=2017,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)

    # When: Evaluated at year 16 (2033-01-15) with historical price €80 in 2025 and €120 in 2033
    res = compute_deemed_disposals(
        test_db,
        evaluation_date=date(2033, 1, 15),
        market_prices={"IE00BFWXDV39": Decimal("120")},
        historical_prices={
            ("IE00BFWXDV39", date(2025, 1, 15)): Decimal("80"),
            ("IE00BFWXDV39", date(2033, 1, 15)): Decimal("120"),
        },
    )

    # Then: 2 deemed disposal events generated
    assert len(res.events) == 2
    # Year 8 event: gain 100 * (80 - 50) = 3000, tax @ 41% = 1230
    assert res.events[0].trigger_date.date() == date(2025, 1, 15)
    assert res.events[0].deemed_gain_eur == Decimal("3000.00")
    assert res.events[0].exit_tax_due_eur == Decimal("1230.00")
    assert res.events[0].stepped_up_cost_per_unit_eur == Decimal("80")

    # Year 16 event: gain 100 * (120 - 80) = 4000 (stepped up to 80), tax @ 38% = 1520
    assert res.events[1].trigger_date.date() == date(2033, 1, 15)
    assert res.events[1].original_cost_basis_eur == Decimal("8000.00")
    assert res.events[1].deemed_gain_eur == Decimal("4000.00")
    assert res.events[1].exit_tax_due_eur == Decimal("1520.00")
    assert res.events[1].stepped_up_cost_per_unit_eur == Decimal("120")


def test_deemed_disposal_open_quantities_only(test_db: DatabaseManager) -> None:
    # Given: ETF classification, 100 units bought in 2017, 60 units sold in 2020.
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="IE00BFWXDV39",
            asset_name="Vanguard UCITS ETF",
            tax_regime=IrishTaxRegime.EXIT_TAX,
            domicile_country="IE",
            is_ucits=True,
        )
    )
    buy = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha1",
        event_timestamp=datetime(2017, 1, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="buy",
        quantity=Decimal("100"),
        unit_price=Decimal("50"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("5000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("5000"),
        tax_year=2017,
        account_country="ireland",
        verification_status="approved",
    )
    sell = FinancialRecord(
        provider="interactive_brokers",
        source_file_sha="sha2",
        event_timestamp=datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="sell",
        quantity=Decimal("60"),
        unit_price=Decimal("60"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("3600"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("3600"),
        tax_year=2020,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)
    insert_financial_record(test_db, sell)

    # When: compute_deemed_disposals called as of 2025-06-01 with price €80
    res = compute_deemed_disposals(
        test_db,
        evaluation_date=date(2025, 6, 1),
        market_prices={"IE00BFWXDV39": Decimal("80")},
    )

    # Then: deemed disposal only applies to remaining 40 open units!
    assert len(res.events) == 1
    event = res.events[0]
    assert event.quantity == Decimal("40")
    assert event.original_cost_basis_eur == Decimal("2000")  # 40 * €50
    assert event.market_value_eur == Decimal("3200")  # 40 * €80
    assert event.deemed_gain_eur == Decimal("1200")  # 3200 - 2000


def test_deemed_disposal_fully_disposed_before_8_years(test_db: DatabaseManager) -> None:
    # Given: 100 units bought in 2017, all 100 units sold in 2020
    test_db.upsert_taxpayer_profile(
        TaxpayerProfile(
            tax_year=2025,
            fiscal_residence_country="IE",
            domicile_country="IE",
            residency_type=ResidencyType.RESIDENT_DOMICILED,
            marginal_tax_rate=Decimal("0.40"),
        )
    )
    test_db.upsert_asset_tax_classification(
        AssetTaxClassification(
            isin="IE00BFWXDV39",
            asset_name="Vanguard UCITS ETF",
            tax_regime=IrishTaxRegime.EXIT_TAX,
            domicile_country="IE",
            is_ucits=True,
        )
    )
    buy = FinancialRecord(
        provider="ibkr",
        source_file_sha="sha1",
        event_timestamp=datetime(2017, 1, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="buy",
        quantity=Decimal("100"),
        unit_price=Decimal("50"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("5000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("5000"),
        tax_year=2017,
        account_country="ireland",
        verification_status="approved",
    )
    sell = FinancialRecord(
        provider="ibkr",
        source_file_sha="sha2",
        event_timestamp=datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc),
        asset_type="etf",
        symbol="VUAA",
        isin="IE00BFWXDV39",
        action="sell",
        quantity=Decimal("100"),
        unit_price=Decimal("60"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("6000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("6000"),
        tax_year=2020,
        account_country="ireland",
        verification_status="approved",
    )
    insert_financial_record(test_db, buy)
    insert_financial_record(test_db, sell)

    # When: Evaluating in 2025 (year 8)
    res = compute_deemed_disposals(
        test_db,
        evaluation_date=date(2025, 6, 1),
        market_prices={"IE00BFWXDV39": Decimal("80")},
    )

    # Then: 0 open units remaining -> 0 deemed disposal events
    assert len(res.events) == 0


def test_get_remitted_amount_includes_post_disposal_remittances(test_db: DatabaseManager):
    # Given: Disposal on 2025-05-10, remittance event on 2025-06-15 (after disposal)
    remittance = RemittanceEvent(
        financial_record_id=100,
        remittance_date=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        amount_eur=Decimal("5000"),
    )
    test_db.add_remittance_event(remittance)

    # When: get_remitted_amount called for financial_record_id 100
    remitted = get_remitted_amount(test_db, 100, tax_year=2025)

    # Then: post-disposal remittance of €5000 is included!
    assert remitted == Decimal("5000")


def test_apply_section581_quarantine_spanning_multiple_repurchase_lots():
    # Given: A loss LotMatch of 20 shares and two repurchase records (Lot 1: 10 shares, Lot 2: 15 shares)
    loss_match = LotMatch(
        source_record_id=1,
        source_account_country="ireland",
        buy_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        matched_quantity=Decimal("20"),
        cost_basis_eur=Decimal("2000.00"),
        proceeds_eur=Decimal("1000.00"),
        gain_loss_eur=Decimal("-1000.00"),
    )
    repurchase_1 = TradeRecord(
        id=10,
        provider="ibkr",
        source_file_sha="sha1",
        event_timestamp=datetime(2025, 2, 1, tzinfo=timezone.utc),
        ingestion_timestamp=datetime(2025, 2, 1, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("10"),
        unit_price=Decimal("100"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("1000"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1000"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )
    repurchase_2 = TradeRecord(
        id=11,
        provider="ibkr",
        source_file_sha="sha2",
        event_timestamp=datetime(2025, 2, 5, tzinfo=timezone.utc),
        ingestion_timestamp=datetime(2025, 2, 5, tzinfo=timezone.utc),
        asset_type=AssetType.STOCK,
        identity=AssetIdentity(symbol="AAPL", isin="US0378331005"),
        action=TransactionAction.BUY,
        quantity=Decimal("15"),
        unit_price=Decimal("105"),
        currency="EUR",
        fees=Decimal("0"),
        total_amount=Decimal("1575"),
        fx_rate=Decimal("1.0"),
        local_total_amount=Decimal("1575"),
        tax_year=2025,
        account_country="ireland",
        verification_status=VerificationStatus.APPROVED,
    )

    # When: apply_section581_quarantine is called
    result = apply_section581_quarantine(
        lot_matches=[loss_match],
        repurchase_records=[repurchase_1, repurchase_2],
    )

    # Then: loss_match splits into 2 chunks attributed to repurchase IDs 10 and 11
    assert len(result) == 2
    assert result[0].matched_quantity == Decimal("10")
    assert result[0].is_section581_restricted is True
    assert result[0].section581_repurchase_record_id == 10
    assert result[0].gain_loss_eur == Decimal("-500.00")

    assert result[1].matched_quantity == Decimal("10")
    assert result[1].is_section581_restricted is True
    assert result[1].section581_repurchase_record_id == 11
    assert result[1].gain_loss_eur == Decimal("-500.00")
