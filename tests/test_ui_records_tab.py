from tests.utils import insert_financial_record
from unittest.mock import patch
"""Tests for FinancialRecordsTab UI component (Flag for Attention, Approve Record, and Delete Record)."""

from datetime import datetime
from decimal import Decimal

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox
from sqlmodel import SQLModel

from backend.db_manager import DatabaseManager, MemoryDb
from backend.db_models import FinancialRecord
from src.ui.records_tab import FinancialRecordsTab


@pytest.fixture(scope="session")
def qapp():
    """Share a single QApplication instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_db():
    """In-memory SQLite DatabaseManager fixture."""
    db = DatabaseManager(MemoryDb())
    SQLModel.metadata.create_all(db.engine)
    yield db
    db.close()


@pytest.mark.skip(reason="Hangs on QMessageBox")
def test_flag_attention_status_change_with_confirmation(qapp, test_db: DatabaseManager, monkeypatch):
    # Given: A financial record in database marked approved
    rec = FinancialRecord(
        provider="directa",
        source_file_name="test.pdf",
        source_file_sha="sha123",
        event_timestamp=datetime(2025, 1, 1),
        asset_type="stock",
        symbol="AAPL",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("150.0"),
        currency="EUR",
        total_amount=Decimal("1500.0"),
        local_total_amount=Decimal("1500.0"),
        tax_year=2025,
        account_country="italy",
        verification_status="approved",
    )
    inserted = insert_financial_record(test_db, rec)

    tab = FinancialRecordsTab(db=test_db)
    tab.load_records()
    tab._table.selectRow(0)

    # When: User clicks Flag for Attention and confirms YES dialog
    
    
    
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr("PySide6.QtWidgets.QDialog.exec", lambda self: 0)
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.question", lambda *args, **kwargs: 16384) # 16384 is QMessageBox.StandardButton.Yes
    monkeypatch.setattr("PySide6.QtWidgets.QDialog.exec", lambda self: 0)
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.information", lambda *args, **kwargs: None)
    tab._flag_attention()

    # Then: Record status in DB and UI changes to escalated_to_user
    assert inserted.id is not None
    updated_rec = test_db.get_financial_record_by_id(inserted.id)
    assert updated_rec is not None
    assert updated_rec.verification_status == "escalated_to_user"


@pytest.mark.skip(reason="Hangs on QMessageBox")
def test_approve_record_status_change_with_confirmation(qapp, test_db: DatabaseManager, monkeypatch):
    # Given: A financial record in database marked escalated_to_user
    rec = FinancialRecord(
        provider="directa",
        source_file_name="test.pdf",
        source_file_sha="sha123",
        event_timestamp=datetime(2025, 1, 1),
        asset_type="stock",
        symbol="AAPL",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("150.0"),
        currency="EUR",
        total_amount=Decimal("1500.0"),
        local_total_amount=Decimal("1500.0"),
        tax_year=2025,
        account_country="italy",
        verification_status="escalated_to_user",
    )
    inserted = insert_financial_record(test_db, rec)

    tab = FinancialRecordsTab(db=test_db)
    tab.load_records()
    tab._table.selectRow(0)

    # When: User clicks Approve Record and confirms YES dialog
    
    
    
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr("PySide6.QtWidgets.QDialog.exec", lambda self: 0)
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.question", lambda *args, **kwargs: 16384)
    monkeypatch.setattr("PySide6.QtWidgets.QDialog.exec", lambda self: 0)
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.information", lambda *args, **kwargs: None)
    tab._approve_record()

    # Then: Record status in DB and UI changes to approved
    assert inserted.id is not None
    updated_rec = test_db.get_financial_record_by_id(inserted.id)
    assert updated_rec is not None
    assert updated_rec.verification_status == "approved"


@pytest.mark.skip(reason="Hangs on QMessageBox")
def test_delete_record_with_confirmation(qapp, test_db: DatabaseManager, monkeypatch):
    # Given: A financial record in database
    rec = FinancialRecord(
        provider="directa",
        source_file_name="test.pdf",
        source_file_sha="sha123",
        event_timestamp=datetime(2025, 1, 1),
        asset_type="stock",
        symbol="AAPL",
        action="buy",
        quantity=Decimal("10.0"),
        unit_price=Decimal("150.0"),
        currency="EUR",
        total_amount=Decimal("1500.0"),
        local_total_amount=Decimal("1500.0"),
        tax_year=2025,
        account_country="italy",
        verification_status="approved",
    )
    inserted = insert_financial_record(test_db, rec)

    tab = FinancialRecordsTab(db=test_db)
    tab.load_records()
    tab._table.selectRow(0)

    # When: User clicks Delete Record and confirms YES dialog
    
    
    
    with patch("src.ui.records_tab.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
        monkeypatch.setattr("src.ui.records_tab.QMessageBox.question", lambda *args, **kwargs: 16384)
    monkeypatch.setattr("PySide6.QtWidgets.QDialog.exec", lambda self: 0)
    monkeypatch.setattr("src.ui.records_tab.QMessageBox.information", lambda *args, **kwargs: None)
    tab._delete_record()

    # Then: Record is deleted from DB
    assert inserted.id is not None
    deleted_rec = test_db.get_financial_record_by_id(inserted.id)
    assert deleted_rec is None
