"""Unit tests for Directa CSV parser."""

from decimal import Decimal
from pathlib import Path

import pytest

from backend.domain_models import TransactionAction
from backend.ingestion.directa_csv_parser import parse_directa_csv


def test_parse_directa_csv_raw_source_fixture():
    # Given: Path to anonymized Directa CSV test fixture copy
    fixture_path = "tests/fixtures/directa_dividendi_anonymized.csv"

    # When: Parsing the test fixture file
    records = parse_directa_csv(fixture_path)

    # Then: Valid StagedFinancialRecord objects are parsed
    assert len(records) == 2

    # Dividend Income Record
    r1 = records[0]
    assert r1.provider == "directa"
    assert r1.action == TransactionAction.DIVIDEND.value
    assert r1.symbol == "REY"
    assert r1.isin == "IT0005282865"
    assert r1.total_amount == Decimal("5.75")
    assert r1.currency == "EUR"
    assert r1.tax_year == 2025
    assert r1.quantity is None
    assert r1.unit_price is None
    assert r1.is_approvable() is True

    # Withholding Tax Record
    r2 = records[1]
    assert r2.action == TransactionAction.TAX_PAYMENT.value
    assert r2.total_amount == Decimal("1.50")
    assert r2.quantity is None
    assert r2.unit_price is None
    assert r2.is_approvable() is True


def test_parse_directa_csv_fails_loud_on_unrecognized_op(tmp_path: Path):
    # Given: Directa CSV with an unsupported operation type
    sample_csv = (
        "Data operazione;Data valuta;Tipo operazione;Ticker;Isin;Protocollo;Descrizione;Quantità;Importo euro;Importo Divisa;Divisa;Riferimento ordine\n"
        "16-05-2025;21-05-2025;Acquisto azioni;REY;IT0005282865;32306290;REPLY;10;150,00;0;EUR;              \n"
    )
    csv_file = tmp_path / "Directa_Unsupported_Op.csv"
    csv_file.write_text(sample_csv, encoding="utf-8")

    # When / Then: Expect ValueError on unsupported operation type
    with pytest.raises(ValueError, match="Unrecognized or unsupported Directa operation type"):
        parse_directa_csv(str(csv_file))
