"""Integration tests for research knowledge ingestion and confidence tracking."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from backend.db_manager import DatabaseManager, LocalDb
from backend.deliberation.models import CourtVerdict, DebateResult, SourceConflict, Traceability
from backend.domain_models import (
    ConfidenceLevel,
    RegulationChunkInput,
    ResearchChunkInput,
    SourceType,
)
from backend.ingestion.chunker import LateChunker
from backend.ingestion.ingest import (
    extract_research_jurisdiction_from_path,
    ingest_research_document,
)


@pytest.fixture
def db_instance(tmp_path: Path):
    db_path = str(tmp_path / "test_db.db")
    vec_path = str(tmp_path / "vector.db")
    db = DatabaseManager(db_config=LocalDb(db_path=db_path, vector_db_path=vec_path))
    yield db
    db.close()


def test_insert_and_retrieve_research_and_regulation_chunks(db_instance: DatabaseManager):
    # Given: A regulation chunk (high confidence) and a research chunk (medium confidence)
    reg_id = db_instance.insert_chunk(
        document_name="tax_law_ireland.pdf",
        jurisdiction="ireland",
        page_number=1,
        text_content="CGT statutory rate is 33 percent under Section 28 TCA 1997.",
        chunk_index=0,
        embedding=[0.1] * 1024,
        document_sha="sha_reg_001",
        source_type=SourceType.REGULATION,
        confidence_level=ConfidenceLevel.HIGH,
    )

    res_id = db_instance.insert_chunk(
        document_name="ai_gemini_research.md",
        jurisdiction=None,
        page_number=0,
        text_content="Irish CGT standard base rate is 33 percent with annual exemption of €1,270.",
        chunk_index=0,
        embedding=[0.1] * 1024,
        document_sha="sha_res_001",
        source_type=SourceType.RESEARCH,
        confidence_level=ConfidenceLevel.MEDIUM,
    )

    # When: Retrieving each chunk by ID
    reg_chunk = db_instance.get_chunk_by_id(reg_id)
    res_chunk = db_instance.get_chunk_by_id(res_id)

    # Then: Attributes, source types, and confidence levels are preserved accurately
    assert reg_chunk is not None
    assert reg_chunk.jurisdiction == "ireland"
    assert reg_chunk.source_type == SourceType.REGULATION
    assert reg_chunk.confidence_level == ConfidenceLevel.HIGH

    assert res_chunk is not None
    assert res_chunk.jurisdiction is None
    assert res_chunk.source_type == SourceType.RESEARCH
    assert res_chunk.confidence_level == ConfidenceLevel.MEDIUM


def test_search_source_type_filtering(db_instance: DatabaseManager):
    # Given: Indexed regulation and research chunks with distinct keywords and embeddings
    _ = db_instance.insert_chunk(
        document_name="revenue_guidance.pdf",
        jurisdiction="ireland",
        page_number=10,
        text_content="Deemed disposal exit tax applies every 8 years at 41 percent.",
        chunk_index=0,
        embedding=[0.5] * 1024,
        document_sha="sha_reg_ireland",
        source_type=SourceType.REGULATION,
        confidence_level=ConfidenceLevel.HIGH,
    )

    _ = db_instance.insert_chunk(
        document_name="perplexity_irish_funds.md",
        jurisdiction="ireland",
        page_number=1,
        text_content="Deemed disposal exit tax under Part 27 is levied at 41 percent.",
        chunk_index=0,
        embedding=[0.5] * 1024,
        document_sha="sha_res_ireland",
        source_type=SourceType.RESEARCH,
        confidence_level=ConfidenceLevel.MEDIUM,
    )

    # When: Executing keyword search with source_type filters
    reg_only = db_instance.keyword_search("deemed disposal", source_type=SourceType.REGULATION)
    res_only = db_instance.keyword_search("deemed disposal", source_type=SourceType.RESEARCH)
    all_results = db_instance.keyword_search("deemed disposal")

    # Then: Filtered results match expected source types
    assert len(reg_only) == 1
    assert reg_only[0].source_type == SourceType.REGULATION
    assert len(res_only) == 1
    assert res_only[0].source_type == SourceType.RESEARCH
    assert len(all_results) == 2

    # When: Executing semantic search with source_type filters
    query_emb = [0.5] * 1024
    sem_reg = db_instance.semantic_search(query_emb, limit=10, source_type=SourceType.REGULATION)
    sem_res = db_instance.semantic_search(query_emb, limit=10, source_type=SourceType.RESEARCH)
    sem_all = db_instance.semantic_search(query_emb, limit=10)

    # Then: Semantic search properly filters by source_type
    assert len(sem_reg) == 1
    assert sem_reg[0].source_type == SourceType.REGULATION
    assert len(sem_res) == 1
    assert sem_res[0].source_type == SourceType.RESEARCH
    assert len(sem_all) == 2


def test_domain_chunk_input_validation():
    # Given / When: Instantiating RegulationChunkInput with and without mandatory jurisdiction
    valid_reg = RegulationChunkInput(
        document_name="official.pdf",
        document_sha="sha1",
        jurisdiction="italy",
        page_number=1,
        text_content="Content",
        chunk_index=0,
        embedding=[0.1] * 1024,
    )
    assert valid_reg.jurisdiction == "italy"
    assert valid_reg.source_type == SourceType.REGULATION
    assert valid_reg.confidence_level == ConfidenceLevel.HIGH

    # Then: Missing jurisdiction in RegulationChunkInput raises ValidationError
    with pytest.raises(ValidationError):
        RegulationChunkInput(  # pyright: ignore[reportCallIssue]
            document_name="official.pdf",
            document_sha="sha1",
            page_number=1,
            text_content="Content",
            chunk_index=0,
            embedding=[0.1] * 1024,
        )

    # Given / When: Instantiating ResearchChunkInput without jurisdiction
    valid_res = ResearchChunkInput(
        document_name="research.md",
        document_sha="sha2",
        page_number=0,
        text_content="Research content",
        chunk_index=0,
        embedding=[0.1] * 1024,
    )

    # Then: ResearchChunkInput allows optional jurisdiction and defaults source_type and confidence
    assert valid_res.jurisdiction is None
    assert valid_res.source_type == SourceType.RESEARCH
    assert valid_res.confidence_level == ConfidenceLevel.MEDIUM


def test_source_conflict_and_debate_result():
    # Given: A detected discrepancy between a regulation source and a research source
    conflict = SourceConflict(
        regulation_source="TCA 1997 Part 19, page 12",
        regulation_claim="CGT rate is 33%",
        research_source="gemini_Irish_tax_Research.md, section 1",
        research_claim="CGT rate is 30%",
        discrepancy_description="Research states outdated CGT rate of 30% instead of statutory 33%.",
    )

    # When: Creating a CourtVerdict containing the conflict
    verdict = CourtVerdict(
        ruling="The statutory rate of 33% applies under Section 28 TCA 1997.",
        traceability=Traceability(source_documents=[]),
        source_conflicts=[conflict],
    )
    debate_result = DebateResult(
        full_transcript="Transcript...",
        verdict=verdict.ruling,
        court_verdict=verdict,
    )

    # Then: DebateResult exposes has_source_conflicts as True
    assert debate_result.has_source_conflicts is True
    assert len(debate_result.court_verdict.source_conflicts) == 1
    assert debate_result.court_verdict.source_conflicts[0].regulation_source == "TCA 1997 Part 19, page 12"

    # Given / When: Verdict without source conflicts
    verdict_clean = CourtVerdict(
        ruling="All sources agree.",
        traceability=Traceability(source_documents=[]),
    )
    clean_result = DebateResult(
        full_transcript="Clean transcript...",
        verdict=verdict_clean.ruling,
        court_verdict=verdict_clean,
    )

    # Then: has_source_conflicts is False
    assert clean_result.has_source_conflicts is False


def test_extract_research_jurisdiction_from_path():
    # Given: Nested research paths and root research paths
    path_nested = "data/research/italy/perplexity_notes.md"
    path_root = "data/research/general_guide.md"

    # When: Extracting inferred jurisdiction
    jur_nested = extract_research_jurisdiction_from_path(path_nested)
    jur_root = extract_research_jurisdiction_from_path(path_root)

    # Then: Inferred jurisdiction matches folder name when nested, or None when root
    assert jur_nested == "italy"
    assert jur_root is None


def test_ingest_research_document_integration(db_instance: DatabaseManager, tmp_path: Path):
    # Given: A markdown research file
    md_content = """# Irish Tax Research
Guidance on capital acquisitions and gains.

## Section 1: CGT Rates
Statutory CGT rate is 33%.

## Section 2: Deductions
Annual CGT exemption is €1,270 per taxpayer.
"""
    file_path = tmp_path / "research_ireland.md"
    file_path.write_text(md_content, encoding="utf-8")

    # Mock chunker to avoid downloading/running heavy model in unit test
    mock_chunker = MagicMock(spec=LateChunker)
    mock_chunk = MagicMock()
    mock_chunk.start_char_idx = 0
    mock_chunk.text_content = "Statutory CGT rate is 33%."
    mock_chunk.parent_text = "Section 1: CGT Rates\nStatutory CGT rate is 33%."
    mock_chunk.embedding = [0.2] * 1024
    mock_chunker.compute_late_chunks.return_value = [mock_chunk]

    # When: Ingesting research document
    ingest_research_document(
        file_path=str(file_path),
        db=db_instance,
        chunker=mock_chunker,
        jurisdiction="ireland",
        confidence=ConfidenceLevel.MEDIUM,
        force=True,
    )

    # Then: Document is recorded in metadata and summary
    ingested_docs = db_instance.get_ingested_documents()
    assert len(ingested_docs) == 1
    assert ingested_docs[0].document_name == "research_ireland.md"
    assert ingested_docs[0].source_type == "research"
    assert ingested_docs[0].jurisdiction == "ireland"

    all_meta = db_instance.get_all_documents_metadata()
    assert len(all_meta) == 1
    assert all_meta[0].source_type == "research"
