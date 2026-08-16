"""Irish CGT orchestrator — bridges DatabaseManager and pure tax computation functions."""

from datetime import date, datetime, timedelta
from decimal import Decimal

from backend.db_manager import DatabaseManager
from backend.domain_models import BaseStrictRecord, TradeRecord, assert_postcondition
from backend.services.accounting.fifo import FIFOAccounting
from src.jurisdiction.ireland.cgt_engine import (
    add_years,
)
from src.jurisdiction.ireland.cgt_engine import (
    compute_deemed_disposals as pure_compute_deemed,
)
from src.jurisdiction.ireland.cgt_engine import (
    compute_disposal as pure_compute_disposal,
)
from src.jurisdiction.ireland.cgt_models import (
    AssetTaxClassification,
    DeemedDisposalResult,
    DisposalInput,
    DisposalResult,
    IrishTaxRegime,
    ResolvedDisposalInput,
    StrictDisposalInput,
    TaxpayerProfile,
)

_SECTION_581_WINDOW_DAYS = 28


def classify_tax_regime(db: DatabaseManager, isin: str) -> AssetTaxClassification:
    """Look up the Irish tax regime classification for a given ISIN.

    Args:
        db: Database manager instance.
        isin: ISIN code of the asset.

    Returns:
        The asset's tax classification record.

    Raises:
        ValueError: If the ISIN is not found in the classification table.
    """
    classification = db.get_asset_tax_classification(isin)
    if classification is None:
        raise ValueError(
            f"ISIN '{isin}' not found in asset_tax_classification table. "
            f"Cannot compute gains without a known tax regime. "
            f"Please add this ISIN to the classification table."
        )
    return classification


def get_taxpayer_profile(db: DatabaseManager, tax_year: int) -> TaxpayerProfile:
    """Retrieve the taxpayer profile for a given tax year.

    Args:
        db: Database manager instance.
        tax_year: The tax year to look up.

    Returns:
        The taxpayer profile record.

    Raises:
        ValueError: If no profile exists for the given tax year.
    """
    profile = db.get_taxpayer_profile(tax_year)
    if profile is None:
        raise ValueError(
            f"No taxpayer profile found for tax year {tax_year}. Please add a profile to the taxpayer_profile table."
        )
    return profile


def check_remittance(db: DatabaseManager, financial_record_id: int) -> bool:
    """Check if disposal proceeds were remitted to Ireland.

    Args:
        db: Database manager instance.
        financial_record_id: Primary key of the financial record.

    Returns:
        True if any remittance event exists for this record.
    """
    events = db.get_remittance_events(financial_record_id)
    return len(events) > 0


def get_remitted_amount(
    db: DatabaseManager,
    financial_record_id: int,
    tax_year: int | None = None,
) -> Decimal:
    """Sum total EUR remitted for a financial record (optionally filtered by tax year).

    Under Irish tax law, remittances of sale proceeds typically occur on or after
    the disposal date. Restricting remittances to dates on or before disposal
    ignores post-sale transfers to Ireland.

    Args:
        db: Database manager instance.
        financial_record_id: Primary key of the financial record.
        tax_year: Optional tax year to scope remittance assessment.

    Returns:
        Total remitted amount in EUR.
    """
    events = db.get_remittance_events(financial_record_id)
    total = Decimal("0")
    for event in events:
        if tax_year is None or event.remittance_date.year == tax_year:
            total += event.amount_eur

    # Post-condition validation
    assert_postcondition(
        total >= Decimal("0"),
        "Negative remitted amount calculated.",
        extra={"financial_record_id": financial_record_id, "tax_year": tax_year, "total": str(total)},
    )
    return total


def load_buy_lots(db: DatabaseManager, isin: str) -> list[TradeRecord]:
    """Load all approved buy records for an ISIN across all jurisdictions.

    Section 580 TCA 1997 treats all shares of the same class as a single
    global pool regardless of which broker or jurisdiction they were
    purchased in. FIFO ordering is by event_timestamp.

    Args:
        db: Database manager instance.
        isin: ISIN code to load buy lots for.

    Returns:
        List of validated TradeRecord buy records ordered by event_timestamp.
    """
    return db.get_validated_buy_records_by_isin(isin)


def detect_section581_repurchase(db: DatabaseManager, isin: str, disposal_date: datetime) -> list[TradeRecord]:
    """Check for repurchases of the same ISIN within 28 days after disposal.

    Section 581(3) TCA 1997: if shares of the same class are sold and
    repurchased within 28 days following disposal, losses are quarantined
    and can only be offset against gains on the repurchased shares.

    Args:
        db: Database manager instance.
        isin: ISIN code for the security.
        disposal_date: Timestamp of the disposal.

    Returns:
        List of validated repurchase TradeRecord instances found in the 28-day window.
    """
    window_end = disposal_date + timedelta(days=_SECTION_581_WINDOW_DAYS)
    return db.get_purchases_in_window(isin, disposal_date, window_end)


def _resolve_disposal_input(db: DatabaseManager, disposal_input: DisposalInput) -> ResolvedDisposalInput:
    """Resolve raw or simulated disposal input into a normalized ResolvedDisposalInput domain model.

    For simulated disposal inputs (hypothetical future sales), the timestamp is set to end-of-day
    (23:59:59) so that buy transactions executed on that same date are correctly matched as preceding
    acquisitions under Section 581(1) TCA 1997 rules.
    """
    if isinstance(disposal_input, StrictDisposalInput):
        raw_record = db.get_financial_record_by_id(disposal_input.record_id)
        if raw_record is None:
            raise ValueError(f"Financial record {disposal_input.record_id} not found.")
        strict_sell = BaseStrictRecord.from_raw(raw_record)
        if not isinstance(strict_sell, TradeRecord):
            raise ValueError(
                f"Financial record {disposal_input.record_id} cannot be "
                f"converted to TradeRecord (missing required fields or invalid action)."
            )
        isin = strict_sell.isin
        if not isin:
            raise ValueError(f"Financial record {disposal_input.record_id} has no ISIN.")

        return ResolvedDisposalInput(
            isin=isin,
            sell_quantity=strict_sell.quantity,
            sell_unit_price_eur=strict_sell.unit_price * strict_sell.fx_rate,
            sell_fees_eur=strict_sell.fees * strict_sell.fx_rate,
            sell_date=strict_sell.event_timestamp,
            disposal_record_id=strict_sell.id,
            asset_name=strict_sell.asset_name,
            is_simulation=False,
        )

    # Simulated disposal input: set timestamp to 23:59:59 for same-day S. 581(1) buy matching
    sell_dt = datetime.combine(disposal_input.disposal_date, datetime.max.time().replace(microsecond=0))
    return ResolvedDisposalInput(
        isin=disposal_input.isin,
        sell_quantity=disposal_input.quantity,
        sell_unit_price_eur=disposal_input.estimated_unit_price_eur,
        sell_fees_eur=disposal_input.estimated_fees_eur,
        sell_date=sell_dt,
        disposal_record_id=None,
        asset_name=None,
        is_simulation=True,
    )


def _get_open_buy_lots(db: DatabaseManager, isin: str, evaluation_date: date) -> dict[int, Decimal]:
    """Calculate remaining open unit quantities per buy lot ID as of evaluation_date.

    Runs a FIFO accounting simulation across all approved buys and sells up to `evaluation_date`.
    Fees are set to 0 because this accounting pass only tracks share unit quantities consumed by prior sales.

    Args:
        db: Database manager instance.
        isin: ISIN code of the asset.
        evaluation_date: Date up to which past sales are evaluated.

    Returns:
        Mapping of buy lot primary key ID to remaining un-sold open share quantity (`dict[int, Decimal]`).
    """
    buy_lots = load_buy_lots(db, isin)
    open_quantities: dict[int, Decimal] = {}
    for lot in buy_lots:
        assert_postcondition(lot.id is not None, "Buy lot primary key ID cannot be None.")
        assert lot.id is not None
        open_quantities[lot.id] = lot.quantity

    validated_sells = db.get_validated_sell_records_by_isin(isin)
    fifo = FIFOAccounting()
    for lot in buy_lots:
        fifo.add_purchase(
            asset=isin,
            acquisition_date=lot.event_timestamp,
            quantity=lot.quantity,
            unit_price=lot.unit_price * lot.fx_rate,
            fees=Decimal("0"),
        )

    for strict_sell in validated_sells:
        if strict_sell.event_timestamp.date() <= evaluation_date:
            raw_matches = fifo.process_sale(
                asset=isin,
                disposal_date=strict_sell.event_timestamp,
                quantity=strict_sell.quantity,
                unit_price=strict_sell.unit_price * strict_sell.fx_rate,
                fees=Decimal("0"),
            )
            for m in raw_matches:
                buy_dt = m.acquisition_date
                matching_lot = next((lot for lot in buy_lots if lot.event_timestamp == buy_dt), None)
                if matching_lot and matching_lot.id in open_quantities:
                    open_quantities[matching_lot.id] = max(Decimal("0"), open_quantities[matching_lot.id] - m.quantity)

    return open_quantities


def compute_disposal(
    db: DatabaseManager,
    disposal_input: DisposalInput,
) -> DisposalResult:
    """Compute the capital gain/loss for a disposal event (DB orchestrated)."""
    resolved = _resolve_disposal_input(db, disposal_input)
    classification = classify_tax_regime(db, resolved.isin)
    regime = IrishTaxRegime(classification.tax_regime)
    profile = get_taxpayer_profile(db, resolved.sell_date.year)
    is_remitted = (
        check_remittance(db, resolved.disposal_record_id) if resolved.disposal_record_id is not None else False
    )
    buy_lots = load_buy_lots(db, resolved.isin)

    cgt_regimes = (IrishTaxRegime.CGT_STANDARD, IrishTaxRegime.OFFSHORE_DISTRIBUTING, IrishTaxRegime.ETC_COMMODITY)
    repurchases = detect_section581_repurchase(db, resolved.isin, resolved.sell_date) if regime in cgt_regimes else []

    remitted_amount = Decimal("0")
    if resolved.disposal_record_id is not None:
        remitted_amount = get_remitted_amount(db, resolved.disposal_record_id, tax_year=resolved.sell_date.year)

    return pure_compute_disposal(
        resolved=resolved,
        buy_lots=buy_lots,
        classification=classification,
        profile=profile,
        repurchase_records=repurchases,
        is_remitted=is_remitted,
        remitted_amount_eur=remitted_amount,
    )


def compute_deemed_disposals(
    db: DatabaseManager,
    evaluation_date: date,
    market_prices: dict[str, Decimal] | None = None,
    historical_prices: dict[tuple[str, date], Decimal] | None = None,
    processed_trigger_dates: set[tuple[int, date]] | None = None,
) -> DeemedDisposalResult:
    """Compute all 8-year deemed disposal events by loading Exit Tax records from database and delegating to the pure CGT engine.

    Args:
        db: Database manager instance.
        evaluation_date: Date to evaluate triggers against.
        market_prices: Optional mapping of ISIN to current EUR unit price.
        historical_prices: Optional mapping of (isin, trigger_date) to EUR unit price on trigger date.
        processed_trigger_dates: Set of (source_record_id, trigger_date) already processed in DB.

    Returns:
        DeemedDisposalResult with all triggered events and aggregated tax liability.
    """
    exit_tax_isins = db.get_isins_by_regime(IrishTaxRegime.EXIT_TAX.value)
    isin_data: list[tuple[str, list[TradeRecord], dict[int, Decimal]]] = []
    required_years: set[int] = set()

    for isin in exit_tax_isins:
        buy_lots = load_buy_lots(db, isin)
        open_quantities = _get_open_buy_lots(db, isin, evaluation_date)
        isin_data.append((isin, buy_lots, open_quantities))

        for lot in buy_lots:
            # Check all 8-year anniversaries up to evaluation_date
            years_held = evaluation_date.year - lot.event_timestamp.year
            for anniversary_multiple in range(1, (years_held // 8) + 1):
                trigger_dt = add_years(lot.event_timestamp, 8 * anniversary_multiple)
                if trigger_dt.date() <= evaluation_date:
                    required_years.add(trigger_dt.year)

    profiles: dict[int, TaxpayerProfile] = {}
    for year in required_years:
        profiles[year] = get_taxpayer_profile(db, year)

    return pure_compute_deemed(
        evaluation_date=evaluation_date,
        isin_data=isin_data,
        taxpayer_profiles=profiles,
        market_prices=market_prices,
        historical_prices=historical_prices,
        processed_trigger_dates=processed_trigger_dates,
    )
