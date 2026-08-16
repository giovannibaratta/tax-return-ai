from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from backend.domain_models import (
    AssetIdentity,
    AssetType,
    BaseStrictRecord,
    DividendRecord,
    TradeRecord,
    TransactionAction,
    VerificationStatus,
)


def build_mock_record(  # noqa: PLR0917
    id: int,
    action: str,
    quantity: Decimal,
    unit_price: Decimal,
    total_amount: Decimal,
    *,
    symbol: str = "UNKNOWN",
    asset_type: str = "stock",
    provider: str = "directa",
    account_country: str = "italy",
    event_timestamp: datetime | None = None,
    isin: str = "US0000000000",
    local_total_amount: Decimal | None = None,
    currency: str = "EUR",
    fx_rate: Decimal = Decimal("1.0"),
    fees: Decimal = Decimal("0.0"),
) -> BaseStrictRecord:
    """Factory to generate realistic, fully-populated strict records directly for testing."""
    if event_timestamp is None:
        event_timestamp = datetime(2025, 1, 15, tzinfo=timezone.utc)
    if local_total_amount is None:
        local_total_amount = total_amount * fx_rate

    action_enum = TransactionAction(action.lower().strip())
    asset_enum = AssetType(asset_type.lower().strip())
    identity = AssetIdentity(symbol=symbol, isin=isin)

    base_kwargs: dict[str, Any] = {
        "id": id,
        "provider": provider,
        "account_country": account_country,
        "event_timestamp": event_timestamp,
        "ingestion_timestamp": event_timestamp,
        "currency": currency,
        "total_amount": total_amount,
        "fx_rate": fx_rate,
        "local_total_amount": local_total_amount,
        "tax_year": event_timestamp.year,
        "verification_status": VerificationStatus.VERIFIED,
    }

    if action_enum in (TransactionAction.BUY, TransactionAction.SELL):
        return TradeRecord(
            **base_kwargs,
            action=action_enum,
            asset_type=asset_enum,
            identity=identity,
            quantity=quantity,
            unit_price=unit_price,
            fees=fees,
        )
    elif action_enum == TransactionAction.DIVIDEND:
        return DividendRecord(**base_kwargs, action=action_enum, asset_type=asset_enum, identity=identity)
    raise NotImplementedError(f"Action {action} not supported in test factory")
