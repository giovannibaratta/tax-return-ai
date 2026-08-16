"""Deterministic CSV parser for Directa SIM transaction and dividend export reports."""

import os
from datetime import datetime, timezone
from decimal import Decimal

from backend.db_models import StagedFinancialRecord
from backend.domain_models import AssetType, TransactionAction
from backend.ingestion.helpers import calculate_sha256


def _clean_decimal(val_str: str) -> Decimal | None:
    """Parse Italian formatted number string into Decimal.

    Italian format uses ',' for decimal separator and '.' for thousands separator.

    Args:
        val_str: Raw formatted string representation of number.

    Returns:
        Parsed Decimal object, or None if input is empty or unparseable.
    """
    if not val_str:
        return None
    s = val_str.strip().replace("€", "").replace(" ", "").replace("$", "")
    if not s:
        return None

    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].replace("-", "").isdigit():
            s = s.replace(".", "")

    try:
        return Decimal(s)
    except Exception:
        return None


def parse_directa_csv(file_path: str) -> list[StagedFinancialRecord]:
    """Parses a Directa SIM export CSV report into StagedFinancialRecord instances.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of populated StagedFinancialRecord objects ready for database staging.

    Raises:
        FileNotFoundError: If the CSV file path does not exist.
        ValueError: If CSV header is missing or row data is invalid/unrecognized.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Directa CSV report file not found: '{file_path}'")

    file_sha = calculate_sha256(file_path)
    file_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Decode text (Directa exports typically use UTF-8 or ISO-8859-1 / Windows-1252)
    try:
        text_content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text_content = file_bytes.decode("latin-1")

    lines = [line for line in text_content.splitlines() if line.strip()]

    # Locate column header row
    header_idx = -1
    for idx, line in enumerate(lines):
        if "Data operazione" in line and "Tipo operazione" in line:
            header_idx = idx
            break

    if header_idx == -1:
        raise ValueError(f"Could not find valid Directa CSV header in file '{file_name}'.")

    header_line = lines[header_idx]
    headers = [col.strip() for col in header_line.split(";")]

    def get_col_idx(col_name: str) -> int:
        """Find index of header column matching col_name exactly (case-insensitive)."""
        for idx, h in enumerate(headers):
            if h.lower().strip() == col_name.lower().strip():
                return idx
        return -1

    idx_date_op = get_col_idx("Data operazione")
    idx_type_op = get_col_idx("Tipo operazione")
    idx_ticker = get_col_idx("Ticker")
    idx_isin = get_col_idx("Isin")
    idx_desc = get_col_idx("Descrizione")
    idx_qty = get_col_idx("Quantità")
    idx_amount_eur = get_col_idx("Importo euro")
    idx_currency = get_col_idx("Divisa")

    records: list[StagedFinancialRecord] = []

    for line_idx in range(header_idx + 1, len(lines)):
        row = [col.strip() for col in lines[line_idx].split(";")]
        if len(row) < len(headers):
            raise ValueError(
                f"Directa CSV row at line {line_idx + 1} has insufficient columns "
                f"(expected {len(headers)}, got {len(row)})."
            )

        raw_date_op = row[idx_date_op] if idx_date_op != -1 and idx_date_op < len(row) else ""
        raw_type_op = row[idx_type_op] if idx_type_op != -1 and idx_type_op < len(row) else ""

        if not raw_date_op or not raw_type_op:
            raise ValueError(f"Directa CSV row at line {line_idx + 1} is missing required date or operation type.")

        # Parse date (DD-MM-YYYY or DD/MM/YYYY or YYYY-MM-DD)
        event_timestamp = None
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                event_timestamp = datetime.strptime(raw_date_op, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                pass

        if event_timestamp is None:
            raise ValueError(f"Directa CSV row at line {line_idx + 1} has unparseable date '{raw_date_op}'.")

        tax_year = event_timestamp.year

        ticker = row[idx_ticker] if idx_ticker != -1 and idx_ticker < len(row) else None
        isin = row[idx_isin] if idx_isin != -1 and idx_isin < len(row) else None
        desc = row[idx_desc] if idx_desc != -1 and idx_desc < len(row) else None
        currency_raw = row[idx_currency] if idx_currency != -1 and idx_currency < len(row) else "EUR"
        currency = currency_raw.upper() if currency_raw else "EUR"

        raw_amount_eur = row[idx_amount_eur] if idx_amount_eur != -1 and idx_amount_eur < len(row) else "0"
        dec_amount_eur = _clean_decimal(raw_amount_eur)

        if dec_amount_eur is None:
            raise ValueError(f"Directa CSV row at line {line_idx + 1} has unparseable EUR amount '{raw_amount_eur}'.")

        # Positive magnitude for total_amount per schema rules
        total_amount = abs(dec_amount_eur)

        # Classify Action and Asset Type
        type_op_lower = raw_type_op.lower()
        if "incasso dividendi" in type_op_lower or "dividendo" in type_op_lower:
            action = TransactionAction.DIVIDEND.value
            asset_type = AssetType.STOCK.value
        elif "ritenuta dividendi" in type_op_lower or "ritenuta" in type_op_lower:
            action = TransactionAction.TAX_PAYMENT.value
            asset_type = AssetType.TAX_PAYMENT.value
        else:
            raise ValueError(
                f"Unrecognized or unsupported Directa operation type '{raw_type_op}' at line {line_idx + 1}."
            )

        # Optional quantity (if present and > 0)
        raw_qty = row[idx_qty] if idx_qty != -1 and idx_qty < len(row) else None
        dec_qty = _clean_decimal(raw_qty) if raw_qty else None
        quantity = dec_qty if (dec_qty is not None and dec_qty != Decimal("0")) else None

        unit_price = None

        record = StagedFinancialRecord(
            provider="directa",
            source_file_name=file_name,
            source_file_sha=file_sha,
            event_timestamp=event_timestamp,
            asset_type=asset_type,
            symbol=ticker.upper() if ticker else None,
            isin=isin.upper() if isin else None,
            asset_name=desc if desc else None,
            action=action,
            quantity=quantity,
            unit_price=unit_price,
            currency=currency,
            fees=Decimal("0.0"),
            total_amount=total_amount,
            fx_rate=Decimal("1.0") if currency == "EUR" else None,
            local_total_amount=total_amount if currency == "EUR" else None,
            tax_year=tax_year,
            account_country="italy",
            verification_status="pending_approval",
        )

        records.append(record)

    return records
