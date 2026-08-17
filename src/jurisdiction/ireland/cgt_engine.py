"""Irish Capital Gains Tax computation engine.

Implements Section 580 FIFO matching, Section 581 bed-and-breakfasting
(preceding 28-day matching priority under S. 581(1) and post-disposal
28-day loss quarantine under S. 581(3)), Exit Tax regime routing,
remittance basis determination, and 8-year deemed disposal scanning per TCA 1997.
"""

import logging
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

from backend.domain_models import (
    StrictFinancialRecord,
    TradeRecord,
    assert_postcondition,
)
from backend.services.accounting.fifo import FIFOAccounting
from src.jurisdiction.ireland.cgt_models import (
    AggregatedLotMatches,
    AssetTaxClassification,
    DeemedDisposalEvent,
    DeemedDisposalResult,
    DisposalResult,
    IrishTaxRegime,
    LotMatch,
    RepurchaseAllocation,
    ResidencyType,
    ResolvedDisposalInput,
    TaxpayerProfile,
)

logger = logging.getLogger(__name__)

# Section 581 TCA 1997: 4-week (28-day) bed-and-breakfasting window
_SECTION_581_WINDOW_DAYS = 28
_FA_2025_CUTOFF_YEAR = 2025


def add_years(dt: datetime, years: int) -> datetime:
    """Add exact calendar years to a datetime, handling leap year Feb 29.

    Args:
        dt: Input datetime.
        years: Number of calendar years to add.

    Returns:
        Datetime shifted by exact calendar years.
    """
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Handle Feb 29 on non-leap year -> Feb 28
        return dt.replace(year=dt.year + years, day=28)


# ---------------------------------------------------------------------------
# Tax regime and rate helpers
# ---------------------------------------------------------------------------


def get_tax_rate(
    regime: IrishTaxRegime,
    event_date: date,
    profile: TaxpayerProfile,
) -> Decimal:
    """Return the applicable tax rate for a regime and event date.

    Args:
        regime: The Irish tax regime of the asset.
        event_date: Date of the chargeable event.
        profile: Mandatory taxpayer profile for the tax year.

    Returns:
        Decimal tax rate (e.g. Decimal('0.33') for 33%).
    """
    rate: Decimal
    if regime == IrishTaxRegime.CGT_STANDARD:
        rate = Decimal("0.33")
    elif regime == IrishTaxRegime.EXIT_TAX:
        # Finance Act 2025: 41% for events <= 2025-12-31, 38% from 2026
        if event_date.year <= _FA_2025_CUTOFF_YEAR:
            rate = Decimal("0.41")
        else:
            rate = Decimal("0.38")
    elif regime == IrishTaxRegime.OFFSHORE_DISTRIBUTING:
        rate = Decimal("0.40")
    elif regime == IrishTaxRegime.OFFSHORE_NON_DISTRIBUTING:
        # Offshore non-distributing: taxed at taxpayer's mandatory marginal income tax rate
        rate = profile.marginal_tax_rate
    elif regime == IrishTaxRegime.ETC_COMMODITY:
        rate = Decimal("0.33")
    else:
        raise ValueError(f"Unknown tax regime: {regime}")

    # Post-condition validation
    assert_postcondition(
        Decimal("0") <= rate <= Decimal("1"),
        "Tax rate out of valid range [0, 1].",
        extra={"regime": regime.value, "event_date": str(event_date), "rate": str(rate)},
    )
    return rate


def is_foreign_asset(classification: AssetTaxClassification) -> bool:
    """Determine if an asset is foreign (non-Irish situated) for remittance basis.

    Args:
        classification: The asset's tax classification record.

    Returns:
        True if the asset is domiciled outside Ireland.

    Raises:
        ValueError: If domicile_country is missing in classification record.
    """
    if classification.domicile_country is None:
        raise ValueError(
            f"ISIN '{classification.isin}' has missing domicile_country in asset_tax_classification table. "
            f"Explicit asset domicile country is required for remittance basis determination."
        )
    return classification.domicile_country.upper() != "IE"


# ---------------------------------------------------------------------------
# Taxpayer profile and taxability
# ---------------------------------------------------------------------------

def determine_irish_taxability(
    profile: TaxpayerProfile,
    classification: AssetTaxClassification,
    *,
    is_remitted: bool,
    is_irish_specified_asset: bool,
) -> tuple[bool, bool]:
    """Determine if a disposal is taxable in Ireland and whether remittance basis applies.

    Statutory Basis:
    - Section 29 & Section 607 TCA 1997: Non-residents are chargeable to Irish CGT
      on Irish specified assets (e.g. Irish land/buildings, minerals, exploration rights,
      assets of an Irish trade/branch, unquoted shares deriving >50% value from Irish land),
      but are NOT chargeable on ordinary quoted shares or foreign assets.
    - Section 747E TCA 1997: Gains on Exit Tax assets (e.g. EU UCITS ETFs under Part 27) are
      taxable for Irish residents regardless of remittance or non-domiciled status.
    - Section 29(4) TCA 1997: Resident non-domiciled individuals are taxable on foreign
      gains on the remittance basis (only to the extent remitted to Ireland).

    Args:
        profile: Taxpayer profile for the disposal year.
        classification: Asset's tax classification.
        is_remitted: Whether proceeds were remitted to Ireland.
        is_irish_specified_asset: Whether the asset is an Irish specified asset under Section 29 TCA 1997.

    Returns:
        Tuple of (taxable_in_ireland, remittance_basis_applies).
    """
    residency = ResidencyType(profile.residency_type)

    if residency == ResidencyType.NON_RESIDENT:
        if is_irish_specified_asset:
            return (True, False)
        return (False, False)

    if residency == ResidencyType.RESIDENT_DOMICILED:
        # Worldwide gains taxable regardless of remittance
        return (True, False)

    # RESIDENT_NON_DOMICILED
    regime = IrishTaxRegime(classification.tax_regime)

    # Exit Tax regime (Part 27 TCA 1997) is taxable regardless of remittance
    if regime == IrishTaxRegime.EXIT_TAX:
        return (True, False)

    if is_irish_specified_asset:
        return (True, False)

    foreign = is_foreign_asset(classification)
    if not foreign:
        # Irish-situated assets always taxable
        return (True, False)

    # Foreign CGT asset under remittance basis
    return (is_remitted, True)


# ---------------------------------------------------------------------------
# FIFO lot loading and matching
# ---------------------------------------------------------------------------


def _process_fifo_chunk(
    lots: list[TradeRecord],
    asset_key: str,
    sell_date: datetime,
    sell_quantity: Decimal,
    sell_unit_price_eur: Decimal,
    *,
    sell_fees_eur: Decimal,
    stepped_up_prices: dict[int, Decimal],
) -> list[LotMatch]:
    """Match a sale quantity against a subset of buy lots using FIFOAccounting.

    Args:
        lots: Chronologically ordered buy records.
        asset_key: Security ISIN or identifier.
        sell_date: Datetime of the sale.
        sell_quantity: Number of units to match.
        sell_unit_price_eur: Sale price per unit in EUR.
        sell_fees_eur: Fees allocated to this sale chunk in EUR.
        stepped_up_prices: Mapping of buy lot ID to stepped-up unit cost basis in EUR
            following Section 747D 8-year Exit Tax deemed disposal triggers.

    Returns:
        List of LotMatch instances.
    """
    fifo = FIFOAccounting()
    for lot in lots:
        if lot.id in stepped_up_prices:
            unit_price_eur = stepped_up_prices[lot.id]
            fees_eur = Decimal("0")
        else:
            unit_price_eur = lot.unit_price * lot.fx_rate
            fees_eur = lot.fees * lot.fx_rate

        fifo.add_purchase(
            asset=asset_key,
            acquisition_date=lot.event_timestamp,
            quantity=lot.quantity,
            unit_price=unit_price_eur,
            fees=fees_eur,
        )

    raw_matches = fifo.process_sale(
        asset=asset_key,
        disposal_date=sell_date,
        quantity=sell_quantity,
        unit_price=sell_unit_price_eur,
        fees=sell_fees_eur,
    )

    lot_matches: list[LotMatch] = []
    for match in raw_matches:
        buy_dt = match.acquisition_date
        source_lot = next((lot for lot in lots if lot.event_timestamp == buy_dt), None)
        assert_postcondition(
            source_lot is not None,
            "Failed to find matching source buy lot by purchase timestamp in FIFO pool.",
            extra={"asset_key": asset_key, "buy_dt": str(buy_dt)},
        )
        assert source_lot is not None
        assert_postcondition(source_lot.id is not None, "Source lot primary key cannot be None.")
        assert source_lot.id is not None

        lot_matches.append(
            LotMatch(
                source_record_id=source_lot.id,
                source_account_country=source_lot.account_country,
                buy_date=buy_dt,
                matched_quantity=match.quantity,
                cost_basis_eur=match.cost_basis,
                proceeds_eur=match.net_proceeds,
                gain_loss_eur=match.realized_gain,
            )
        )
    return lot_matches


def _partition_lots(buy_lots: list[TradeRecord], sell_date: datetime) -> tuple[list[TradeRecord], list[TradeRecord]]:
    """Partition buy lots into Section 581(1) preceding 28-day lots and older FIFO lots.

    Statutory Basis:
    - Section 581(1) TCA 1997: Shares acquired within 4 weeks preceding disposal are identified first.
    - Section 581(2) TCA 1997: Within the 4-week window, shares acquired at an earlier date are identified
      before shares acquired at a later date (chronological / FIFO order within the 28-day window).
    """
    preceding_lots: list[TradeRecord] = []
    older_lots: list[TradeRecord] = []

    # sell_d is retained to compute the exact calendar day delta required by Section 581's 28-day window rule,
    # as the statutory window relies on calendar dates rather than an exact 672-hour sliding window.
    sell_d = sell_date.date()

    for lot in buy_lots:
        if lot.event_timestamp <= sell_date:
            lot_d = lot.event_timestamp.date()
            days_diff = (sell_d - lot_d).days
            if days_diff <= _SECTION_581_WINDOW_DAYS:
                preceding_lots.append(lot)
            else:
                older_lots.append(lot)

    # Both partitions ordered chronologically per Section 581(2) and Section 580 TCA 1997
    preceding_lots.sort(key=lambda r: r.event_timestamp)
    older_lots.sort(key=lambda r: r.event_timestamp)
    return preceding_lots, older_lots


def match_lots_fifo(
    buy_lots: list[TradeRecord],
    sell_quantity: Decimal,
    sell_unit_price_eur: Decimal,
    sell_fees_eur: Decimal,
    sell_date: datetime,
    *,
    stepped_up_unit_prices: dict[int, Decimal] | None = None,
) -> list[LotMatch]:
    """Match a sell against buy lots using Section 581(1)/(2) 28-day preceding priority and Section 580 FIFO.

    Statutory Basis:
    - Section 581(1) TCA 1997: Shares acquired within the 4 weeks (28 days) preceding disposal
      are identified first before matching against older holdings.
    - Section 581(2) TCA 1997: Within the 4-week window, shares acquired at an earlier date
      are identified first (FIFO within the 28-day window).
    - Section 580 TCA 1997: Remaining shares are identified on a First-In, First-Out (FIFO) basis.

    Args:
        buy_lots: Chronologically ordered buy records.
        sell_quantity: Number of units being sold.
        sell_unit_price_eur: Sale price per unit in EUR.
        sell_fees_eur: Total sale fees in EUR.
        sell_date: Datetime of the sale.
        stepped_up_unit_prices: Mapping of buy lot record ID to stepped-up unit price in EUR.

    Returns:
        List of LotMatch results, one per consumed (or partially consumed) lot.

    Raises:
        ValueError: If insufficient buy lots to cover the sell quantity or if ISIN is missing.
    """
    if not buy_lots:
        raise ValueError("No buy lots available for FIFO matching.")

    asset_key = buy_lots[0].isin
    if not asset_key:
        raise ValueError(f"Buy lot ID {buy_lots[0].id} has missing ISIN.")

    stepped_up_prices = stepped_up_unit_prices or {}
    preceding_lots, older_lots = _partition_lots(buy_lots, sell_date)

    remaining_sell_qty = sell_quantity
    lot_matches: list[LotMatch] = []

    # Priority 1: Match against Section 581(1)/(2) preceding 28-day acquisitions (earlier first per S.581(2))
    if preceding_lots and remaining_sell_qty > Decimal("0"):
        preceding_available = sum((lot.quantity for lot in preceding_lots), Decimal("0"))
        preceding_match_qty = min(remaining_sell_qty, preceding_available)
        preceding_fees = (
            sell_fees_eur * (preceding_match_qty / sell_quantity) if sell_quantity > Decimal("0") else Decimal("0")
        )
        preceding_matches = _process_fifo_chunk(
            lots=preceding_lots,
            asset_key=asset_key,
            sell_date=sell_date,
            sell_quantity=preceding_match_qty,
            sell_unit_price_eur=sell_unit_price_eur,
            sell_fees_eur=preceding_fees,
            stepped_up_prices=stepped_up_prices,
        )
        lot_matches.extend(preceding_matches)
        for m in preceding_matches:
            remaining_sell_qty -= m.matched_quantity

    # Priority 2: Match remaining quantity against older holdings under Section 580 FIFO rules
    if older_lots and remaining_sell_qty > Decimal("0"):
        older_fees = sell_fees_eur * (remaining_sell_qty / sell_quantity) if sell_quantity > Decimal("0") else Decimal("0")
        older_matches = _process_fifo_chunk(
            lots=older_lots,
            asset_key=asset_key,
            sell_date=sell_date,
            sell_quantity=remaining_sell_qty,
            sell_unit_price_eur=sell_unit_price_eur,
            sell_fees_eur=older_fees,
            stepped_up_prices=stepped_up_prices,
        )
        lot_matches.extend(older_matches)
        for m in older_matches:
            remaining_sell_qty -= m.matched_quantity

    if remaining_sell_qty > Decimal("0"):
        raise ValueError(
            f"Insufficient buy lots for ISIN '{asset_key}'. "
            f"Required {sell_quantity}, remaining unallocated {remaining_sell_qty} units."
        )

    # Post-condition validations
    total_matched = sum((m.matched_quantity for m in lot_matches), Decimal("0"))
    assert_postcondition(
        total_matched == sell_quantity,
        "Total matched quantity does not match requested sell quantity.",
        extra={"sell_quantity": str(sell_quantity), "total_matched": str(total_matched), "asset_key": asset_key},
    )

    for m in lot_matches:
        assert_postcondition(
            m.matched_quantity > Decimal("0"),
            "Non-positive matched lot quantity.",
            extra={"source_record_id": m.source_record_id, "matched_quantity": str(m.matched_quantity)},
        )
        assert_postcondition(
            m.cost_basis_eur >= Decimal("0"),
            "Negative cost basis for matched lot.",
            extra={"source_record_id": m.source_record_id, "cost_basis_eur": str(m.cost_basis_eur)},
        )

    return lot_matches


def _quarantine_match_chunk(
    match: LotMatch,
    repurchase_allocations: list[RepurchaseAllocation],
) -> tuple[list[LotMatch], list[RepurchaseAllocation]]:
    """Helper to split or restrict a LotMatch under Section 581(3) TCA 1997.

    If a sale realized a capital loss and shares of the same ISIN were repurchased
    within 28 days post-sale, Section 581(3) restricts (quarantines) the loss up to
    the repurchased quantity, attributing each restricted loss chunk to its specific
    repurchase record ID. Spanning across multiple repurchase lots produces separate
    attributed LotMatch chunks without object mutation.
    """
    unprocessed_qty: Decimal = match.matched_quantity
    unit_gain_loss: Decimal = match.gain_loss_eur / match.matched_quantity
    unit_cost: Decimal = match.cost_basis_eur / match.matched_quantity
    unit_proceeds: Decimal = match.proceeds_eur / match.matched_quantity

    allocs: list[RepurchaseAllocation] = list(repurchase_allocations)
    res_chunks: list[LotMatch] = []

    while unprocessed_qty > Decimal("0"):
        if allocs:
            alloc: RepurchaseAllocation = allocs[0]
            take_qty = min(unprocessed_qty, alloc.remaining_qty)

            res_chunks.append(
                LotMatch(
                    source_record_id=match.source_record_id,
                    source_account_country=match.source_account_country,
                    buy_date=match.buy_date,
                    matched_quantity=take_qty,
                    cost_basis_eur=unit_cost * take_qty,
                    proceeds_eur=unit_proceeds * take_qty,
                    gain_loss_eur=unit_gain_loss * take_qty,
                    is_section581_restricted=True,
                    section581_repurchase_record_id=alloc.record_id,
                )
            )

            unprocessed_qty -= take_qty
            rem_qty = alloc.remaining_qty - take_qty
            rest_allocs: list[RepurchaseAllocation] = allocs[1:]
            if rem_qty <= Decimal("0"):
                allocs = rest_allocs
            else:
                updated_head: RepurchaseAllocation = alloc.model_copy(update={"remaining_qty": rem_qty})
                allocs = [updated_head, *rest_allocs]
        else:
            # No repurchase capacity left -> remaining chunk is unrestricted
            res_chunks.append(
                LotMatch(
                    source_record_id=match.source_record_id,
                    source_account_country=match.source_account_country,
                    buy_date=match.buy_date,
                    matched_quantity=unprocessed_qty,
                    cost_basis_eur=unit_cost * unprocessed_qty,
                    proceeds_eur=unit_proceeds * unprocessed_qty,
                    gain_loss_eur=unit_gain_loss * unprocessed_qty,
                    is_section581_restricted=False,
                    section581_repurchase_record_id=None,
                )
            )
            unprocessed_qty = Decimal("0")

    return res_chunks, allocs


def apply_section581_quarantine(
    lot_matches: list[LotMatch],
    repurchase_records: Sequence[StrictFinancialRecord] | None,
) -> list[LotMatch]:
    """Apply Section 581(3) loss quarantine proportionally up to repurchased quantity.

    Section 581(3) TCA 1997 anti-avoidance rule: Capital losses realized on a share disposal
    followed by a repurchase of the same ISIN within 28 days post-sale are quarantined.
    The quarantined loss cannot offset general capital gains; it can ONLY offset future gains
    when those specific repurchased shares are eventually disposed of.
    """
    if not repurchase_records:
        return lot_matches

    repurchase_allocations: list[RepurchaseAllocation] = []
    for r in repurchase_records:
        if isinstance(r, TradeRecord):
            if r.id is None:
                raise ValueError(f"Repurchase record has missing primary key ID: {r}")
            if r.quantity <= Decimal("0"):
                raise ValueError(f"Repurchase record {r.id} has non-positive quantity: {r.quantity}")
            repurchase_allocations.append(RepurchaseAllocation(record_id=r.id, remaining_qty=r.quantity))

    total_repurchased_qty: Decimal = sum((alloc.remaining_qty for alloc in repurchase_allocations), Decimal("0"))
    if total_repurchased_qty <= Decimal("0"):
        return lot_matches

    updated: list[LotMatch] = []

    for match in lot_matches:
        has_quarantine_capacity = sum((a.remaining_qty for a in repurchase_allocations), Decimal("0")) > Decimal("0")
        if match.gain_loss_eur < Decimal("0") and has_quarantine_capacity:
            chunks, repurchase_allocations = _quarantine_match_chunk(match, repurchase_allocations)
            updated.extend(chunks)
        else:
            updated.append(match)

    # Post-condition validation
    total_updated_qty = sum((m.matched_quantity for m in updated), Decimal("0"))
    total_original_qty = sum((m.matched_quantity for m in lot_matches), Decimal("0"))
    assert_postcondition(
        total_updated_qty == total_original_qty,
        "Quantity mismatch after Section 581 loss quarantine.",
        extra={"total_updated_qty": str(total_updated_qty), "total_original_qty": str(total_original_qty)},
    )

    return updated


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _aggregate_lot_matches(lot_matches: list[LotMatch]) -> AggregatedLotMatches:
    """Aggregate total proceeds, cost basis, gross gain, restricted losses, and unrestricted gain."""
    total_proceeds = Decimal("0")
    total_cost_basis = Decimal("0")
    gross_gain_loss = Decimal("0")
    restricted_losses = Decimal("0")
    unrestricted_gain_loss = Decimal("0")

    for m in lot_matches:
        total_proceeds += m.proceeds_eur
        total_cost_basis += m.cost_basis_eur
        gross_gain_loss += m.gain_loss_eur
        if m.is_section581_restricted:
            restricted_losses += abs(m.gain_loss_eur)
        else:
            unrestricted_gain_loss += m.gain_loss_eur

    totals = AggregatedLotMatches(
        total_proceeds_eur=total_proceeds,
        total_cost_basis_eur=total_cost_basis,
        gross_gain_loss_eur=gross_gain_loss,
        section581_quarantined_loss_eur=restricted_losses,
        unrestricted_gain_loss_eur=unrestricted_gain_loss,
    )

    # Post-condition validation
    assert_postcondition(
        totals.gross_gain_loss_eur == totals.total_proceeds_eur - totals.total_cost_basis_eur,
        "Gross gain/loss does not equal total proceeds minus total cost basis.",
        extra={"proceeds": str(totals.total_proceeds_eur), "cost_basis": str(totals.total_cost_basis_eur)},
    )
    return totals


def compute_disposal(
    resolved: ResolvedDisposalInput,
    buy_lots: list[TradeRecord],
    classification: AssetTaxClassification,
    profile: TaxpayerProfile,
    repurchase_records: list[TradeRecord] | None = None,
    is_remitted: bool = False,
    remitted_amount_eur: Decimal = Decimal("0"),
    stepped_up_unit_prices: dict[int, Decimal] | None = None,
) -> DisposalResult:
    """Compute the capital gain/loss for a disposal event.

    Args:
        resolved: The resolved disposal input.
        buy_lots: All available buy lots for matching.
        classification: The asset's tax classification.
        profile: The taxpayer's profile for the year.
        repurchase_records: Post-disposal repurchase lots (if applicable for Section 581).
        is_remitted: Whether the proceeds were remitted to Ireland.
        remitted_amount_eur: The amount remitted in EUR.
        stepped_up_unit_prices: Optional map of stepped-up unit prices.

    Returns:
        Complete DisposalResult with gain breakdown, lot matches, and tax flags.
    """
    regime = IrishTaxRegime(classification.tax_regime)
    asset_name = resolved.asset_name or classification.asset_name

    taxable_in_ireland, remittance_basis_applies = determine_irish_taxability(
        profile, classification, is_remitted=is_remitted, is_irish_specified_asset=False
    )

    # Load buy lots & match via Section 581(1) 28-day preceding priority + FIFO
    lot_matches = match_lots_fifo(
        buy_lots=buy_lots,
        sell_quantity=resolved.sell_quantity,
        sell_unit_price_eur=resolved.sell_unit_price_eur,
        sell_fees_eur=resolved.sell_fees_eur,
        sell_date=resolved.sell_date,
    )

    # Section 581(3) post-disposal repurchases & quarantine
    cgt_regimes = (IrishTaxRegime.CGT_STANDARD, IrishTaxRegime.OFFSHORE_DISTRIBUTING, IrishTaxRegime.ETC_COMMODITY)
    repurchases = repurchase_records if (repurchase_records and regime in cgt_regimes) else []
    lot_matches = apply_section581_quarantine(lot_matches, repurchases)

    # Aggregate match results
    totals = _aggregate_lot_matches(lot_matches)

    unrestricted_gain_loss = totals.unrestricted_gain_loss_eur

    # Remittance basis rules (Section 29 TCA 1997)
    if remittance_basis_applies and taxable_in_ireland:
        if unrestricted_gain_loss < Decimal("0"):
            # Capital losses on unremitted foreign assets are completely unallowable under remittance basis
            unrestricted_gain_loss = Decimal("0")
        elif resolved.disposal_record_id is not None and totals.total_proceeds_eur > Decimal("0"):
            # Remittance basis: Taxable gain is capped at actual remitted EUR amount
            unrestricted_gain_loss = min(unrestricted_gain_loss, remitted_amount_eur)

    tax_rate = get_tax_rate(
        regime, resolved.sell_date.date() if isinstance(resolved.sell_date, datetime) else resolved.sell_date, profile
    )

    result = DisposalResult(
        disposal_record_id=resolved.disposal_record_id,
        isin=resolved.isin,
        asset_name=asset_name,
        tax_regime=regime,
        disposal_date=resolved.sell_date,
        total_quantity=resolved.sell_quantity,
        total_proceeds_eur=totals.total_proceeds_eur,
        total_cost_basis_eur=totals.total_cost_basis_eur,
        gross_gain_loss_eur=totals.gross_gain_loss_eur,
        unrestricted_gain_loss_eur=unrestricted_gain_loss,
        section581_quarantined_loss_eur=totals.section581_quarantined_loss_eur,
        applicable_tax_rate=tax_rate,
        annual_exemption_applicable=regime in cgt_regimes,
        loss_offset_allowed=regime in cgt_regimes,
        deemed_disposal_applies=regime == IrishTaxRegime.EXIT_TAX,
        remittance_basis_applies=remittance_basis_applies,
        taxable_in_ireland=taxable_in_ireland,
        lot_matches=lot_matches,
        is_simulation=resolved.is_simulation,
    )

    # Post-condition validations
    assert_postcondition(
        result.total_quantity > Decimal("0"),
        "Non-positive total disposal quantity.",
        extra={"isin": resolved.isin, "total_quantity": str(result.total_quantity)},
    )
    assert_postcondition(
        result.total_cost_basis_eur >= Decimal("0"),
        "Negative total cost basis in disposal result.",
        extra={"isin": resolved.isin, "total_cost_basis_eur": str(result.total_cost_basis_eur)},
    )
    assert_postcondition(
        Decimal("0") <= result.applicable_tax_rate <= Decimal("1"),
        "Applicable tax rate out of valid range [0, 1].",
        extra={"isin": resolved.isin, "applicable_tax_rate": str(result.applicable_tax_rate)},
    )

    return result


# ---------------------------------------------------------------------------
# Deemed disposal scanner
# ---------------------------------------------------------------------------


def compute_deemed_disposals(
    evaluation_date: date,
    isin_data: list[tuple[str, list[TradeRecord], dict[int, Decimal]]],
    taxpayer_profiles: dict[int, TaxpayerProfile],
    market_prices: dict[str, Decimal] | None = None,
    historical_prices: dict[tuple[str, date], Decimal] | None = None,
    processed_trigger_dates: set[tuple[int, date]] | None = None,
) -> DeemedDisposalResult:
    """Pure in-memory computation scanning open Exit Tax lots for rolling 8-year deemed disposal triggers.

    Per Section 747D TCA 1997, UCITS ETF lots are deemed disposed on their
    8th, 16th, 24th... anniversaries at fair market value on the exact trigger date.
    The cost basis is stepped up and the 8-year clock resets.

    Args:
        evaluation_date: Date to evaluate triggers against.
        isin_data: List of tuples containing (isin, buy_lots, open_quantities_by_lot_id).
        taxpayer_profiles: Mapping of tax year to TaxpayerProfile.
        market_prices: Optional mapping of ISIN to current EUR unit price.
        historical_prices: Optional mapping of (isin, trigger_date) to EUR unit price on trigger date.
        processed_trigger_dates: Set of (source_record_id, trigger_date) already processed.

    Returns:
        DeemedDisposalResult with all triggered events and aggregated tax.

    Raises:
        ValueError: If market price is missing for a triggered deemed disposal date.
    """
    events: list[DeemedDisposalEvent] = []
    total_gain = Decimal("0")
    total_tax = Decimal("0")

    prices = market_prices or {}
    hist_prices = historical_prices or {}
    already_processed = processed_trigger_dates or set()

    # Get all ISINs classified as exit_tax

    for isin, buy_lots, open_quantities in isin_data:
        for lot in buy_lots:
            assert_postcondition(lot.id is not None, "Buy lot ID cannot be None in deemed disposal scanner.")
            assert lot.id is not None

            assert_postcondition(
                lot.id in open_quantities,
                "Buy lot missing from open quantity map.",
                extra={"lot_id": lot.id, "isin": isin},
            )
            open_qty = open_quantities[lot.id]
            if open_qty <= Decimal("0"):
                continue

            current_cost_basis = open_qty * lot.unit_price * lot.fx_rate
            acquisition_dt = lot.event_timestamp

            # Rolling 8-year anniversaries (8, 16, 24 years...)
            years_offset = 8
            while True:
                trigger_dt = add_years(acquisition_dt, years_offset)
                trigger_d = trigger_dt.date()

                if trigger_d > evaluation_date:
                    break

                years_offset += 8

                # Skip if this trigger date for this lot was already processed
                if (lot.id, trigger_d) in already_processed:
                    continue

                price_on_trigger = hist_prices.get((isin, trigger_d)) or prices.get(isin)
                if price_on_trigger is None:
                    raise ValueError(
                        f"Missing market price for ISIN '{isin}' on 8-year deemed disposal trigger date {trigger_d}. "
                        f"Cannot compute exit tax without an explicit fair market value price."
                    )

                market_value = open_qty * price_on_trigger
                cost_basis = current_cost_basis
                deemed_gain = max(Decimal("0"), market_value - cost_basis)

                profile = taxpayer_profiles.get(trigger_d.year)
                if not profile:
                    raise ValueError(f"Missing profile for {trigger_d.year}")
                rate = get_tax_rate(IrishTaxRegime.EXIT_TAX, trigger_d, profile)
                tax_due = deemed_gain * rate

                stepped_up_cost = price_on_trigger

                events.append(
                    DeemedDisposalEvent(
                        source_record_id=lot.id,
                        isin=isin,
                        acquisition_date=acquisition_dt,
                        trigger_date=trigger_dt,
                        quantity=open_qty,
                        original_cost_basis_eur=cost_basis,
                        market_value_eur=market_value,
                        deemed_gain_eur=deemed_gain,
                        exit_tax_rate=rate,
                        exit_tax_due_eur=tax_due,
                        stepped_up_cost_per_unit_eur=stepped_up_cost,
                    )
                )

                total_gain += deemed_gain
                total_tax += tax_due

                # Update running cost basis for subsequent rolling anniversaries
                current_cost_basis = open_qty * stepped_up_cost

    result = DeemedDisposalResult(
        evaluation_date=evaluation_date,
        events=events,
        total_deemed_gain_eur=total_gain,
        total_exit_tax_due_eur=total_tax,
    )

    # Post-condition validations
    assert_postcondition(
        result.total_deemed_gain_eur >= Decimal("0"),
        "Negative total deemed gain in deemed disposal result.",
        extra={"evaluation_date": str(evaluation_date), "total_deemed_gain_eur": str(result.total_deemed_gain_eur)},
    )
    assert_postcondition(
        result.total_exit_tax_due_eur >= Decimal("0"),
        "Negative total exit tax due in deemed disposal result.",
        extra={"evaluation_date": str(evaluation_date), "total_exit_tax_due_eur": str(result.total_exit_tax_due_eur)},
    )

    return result
