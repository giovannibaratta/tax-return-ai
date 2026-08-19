"""Tests for shared tax tools in backend/tools/tax_tools.py."""

from __future__ import annotations

import asyncio
import os
import tempfile
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from sqlmodel import Session

from backend.chat.agent import ChatDeps
from backend.db_manager import DatabaseManager, MemoryDb, TaxDocumentMetadata
from backend.ingestion.exporter import export_pages_to_disk
from backend.tools.tax_tools import (
    calculate_action,
    get_chunk_action,
    get_chunk_neighbors_action,
    get_taxpayer_profile_action,
    list_documents_action,
    read_doc_page_action,
    register_tax_tools,
)
from src.jurisdiction.ireland.cgt_models import ResidencyType, TaxpayerProfile


def test_calculate_action() -> None:
    # Given: arithmetic mathematical expressions
    # When: evaluated via calculate_action
    res1 = calculate_action("3500 * 0.26")
    res2 = calculate_action("12500 * 0.41 - 1270")

    # Then: accurate Decimal results returned
    assert res1 == Decimal("910.0")
    assert res2 == Decimal("3855.0")


def test_calculate_precision() -> None:
    # Given: an expression that would cause float precision issues (0.1 + 0.2)
    # When: evaluated via calculate_action
    res = calculate_action("0.1 + 0.2")

    # Then: it exactly equals Decimal("0.3"), proving Decimal math is used during evaluation
    assert res == Decimal("0.3")


def test_tax_tools_db_actions() -> None:
    # Given: a database with ingested chunks
    db = DatabaseManager(MemoryDb())

    with Session(db.engine) as session:
        doc1 = TaxDocumentMetadata(
            id=1,
            document_name="guide.pdf",
            document_sha="sha_abc",
            jurisdiction="ireland",
            source_type="regulation",
            confidence_level="high",
            page_number=1,
            chunk_index=0,
            text_content="Page 1 Content",
        )
        doc2 = TaxDocumentMetadata(
            id=2,
            document_name="guide.pdf",
            document_sha="sha_abc",
            jurisdiction="ireland",
            source_type="regulation",
            confidence_level="high",
            page_number=2,
            chunk_index=1,
            text_content="Page 2 Content",
        )
        session.add_all([doc1, doc2])
        session.commit()
        target_id = doc1.id
        assert target_id is not None

    # When: executing list_documents_action and get_chunk_action
    docs = list_documents_action(db)
    chunk = get_chunk_action(db, target_id)
    neighbors = get_chunk_neighbors_action(db, target_id, window=1)

    # Then: metadata is returned correctly
    assert len(docs) == 1
    assert docs[0].document_name == "guide.pdf"

    assert chunk is not None
    assert chunk.id == target_id

    assert len(neighbors) == 2


def test_register_tax_tools() -> None:
    # Given: a PydanticAI test agent
    test_model = TestModel()
    agent = Agent(model=test_model, deps_type=ChatDeps)

    # When: registering tax tools
    register_tax_tools(agent)

    # Then: tools are registered successfully on agent
    toolset = getattr(agent, "_function_toolset", None)
    tool_names: set[str] = set(getattr(toolset, "tools", {}).keys())
    assert "query_tax_knowledge" in tool_names
    assert "list_documents" in tool_names
    assert "get_chunk" in tool_names
    assert "get_chunk_neighbors" in tool_names
    assert "read_doc_page" in tool_names
    assert "calculate" in tool_names
    assert "get_financial_record" in tool_names
    assert "filter_financial_records" in tool_names
    assert "get_tax_income_records" in tool_names


def test_get_taxpayer_profile_action() -> None:
    # Given: a database with taxpayer profiles inserted
    db = DatabaseManager(MemoryDb())

    profile = TaxpayerProfile(
        tax_year=2025,
        fiscal_residence_country="IE",
        domicile_country="IT",
        residency_type=ResidencyType.RESIDENT_NON_DOMICILED,
        marginal_tax_rate=Decimal("0.40"),
    )
    db.upsert_taxpayer_profile(profile)

    # When: get_taxpayer_profile_action is invoked for tax year 2025
    profiles = get_taxpayer_profile_action(db, tax_year=2025)

    # Then: profile contains expected fields
    assert len(profiles) == 1
    assert profiles[0].tax_year == 2025
    assert profiles[0].fiscal_residence_country == "IE"
    assert profiles[0].domicile_country == "IT"
    assert profiles[0].residency_type == ResidencyType.RESIDENT_NON_DOMICILED


def test_read_doc_page_tool() -> None:
    # Given: A database with chunks across multiple pages
    db = DatabaseManager(MemoryDb())

    with Session(db.engine) as session:
        c1 = TaxDocumentMetadata(
            id=10,
            document_name="irish_cgt_guide.pdf",
            document_sha="sha_cgt",
            jurisdiction="ireland",
            source_type="regulation",
            confidence_level="high",
            page_number=3,
            chunk_index=0,
            text_content="Page 3 Part 1: Deemed disposal 8 year rule applies to UCITS ETFs.",
        )
        c2 = TaxDocumentMetadata(
            id=11,
            document_name="irish_cgt_guide.pdf",
            document_sha="sha_cgt",
            jurisdiction="ireland",
            source_type="regulation",
            confidence_level="high",
            page_number=3,
            chunk_index=1,
            text_content="Page 3 Part 2: Exit tax rate is 41%.",
        )
        session.add_all([c1, c2])
        session.commit()

    # When: Reading document page 3 via tool action
    page = read_doc_page_action(db=db, document_name="irish_cgt_guide.pdf", page_number=3)

    # Then: Full page text is concatenated
    assert page is not None
    assert page.page_number == 3
    assert "Deemed disposal 8 year rule" in page.text_content
    assert "Exit tax rate is 41%" in page.text_content
