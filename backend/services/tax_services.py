"""Standalone reusable tax services."""

from datetime import datetime, time
from decimal import Decimal

from backend.db_manager import DatabaseManager
from backend.deliberation.models import EvidenceChunk
from backend.domain_models import BaseStrictRecord, DocumentMetadata, DocumentPageInfo, StrictTaxIncomeRecord
from backend.llm.embedding_runner import BaseEmbeddingRunner, BgeM3EmbeddingRunner
from backend.llm.reranker import BgeCrossEncoderReranker
from backend.utils.math_utils import evaluate_expression
from src.jurisdiction.ireland.cgt_models import TaxpayerProfile


def query_tax_knowledge_action(  # noqa: PLR0917
    db: DatabaseManager,
    embedding_runner: BaseEmbeddingRunner | None,
    query_text: str,
    limit: int = 5,
    jurisdiction: str | None = None,
    reranker: BgeCrossEncoderReranker | None = None,
) -> list[EvidenceChunk]:
    """Execute two-stage RAG search (Vector retrieval + Cross-Encoder Reranking).

    Args:
        db: DatabaseManager instance.
        embedding_runner: BaseEmbeddingRunner instance or None.
        query_text: Natural language query string.
        limit: Max results to return after reranking.
        jurisdiction: Optional jurisdiction filter.
        reranker: BgeCrossEncoderReranker instance or None.

    Returns:
        List of EvidenceChunk.
    """
    if embedding_runner is None:
        embedding_runner = BgeM3EmbeddingRunner()

    query_embedding = embedding_runner.embed(query_text)

    # Stage 1: Hybrid Candidate Retrieval (Dense Vector + Keyword Over-fetching)
    candidate_pool_size = max(limit * 20, 150)
    vec_candidates = db.semantic_search(
        query_embedding=query_embedding,
        limit=candidate_pool_size,
        jurisdiction=jurisdiction,
    )

    kw_candidates = db.keyword_search(
        query_text=query_text,
        limit=50,
        jurisdiction=jurisdiction,
    )

    candidate_map: dict[int, EvidenceChunk] = {chunk.id: chunk for chunk in vec_candidates}
    for kw_chunk in kw_candidates:
        if kw_chunk.id not in candidate_map:
            candidate_map[kw_chunk.id] = kw_chunk

    # Stage 2: Cross-Encoder Reranking
    result_chunks: list[EvidenceChunk] = []
    if candidate_map:
        if reranker is None:
            reranker = BgeCrossEncoderReranker()
        reranked_results = reranker.rerank(
            query=query_text,
            candidates=list(candidate_map.values()),
            text_extractor=lambda x: x.text_content,
            top_k=limit,
        )
        for res in reranked_results:
            item_dict = res.item.model_dump()
            item_dict["distance"] = res.rerank_score  # Using distance field for rerank score for simplicity
            result_chunks.append(EvidenceChunk(**item_dict))

    return result_chunks


def list_documents_action(
    db: DatabaseManager,
) -> list[DocumentMetadata]:
    """List all tax regulatory documents and manuals in database.

    Args:
        db: DatabaseManager instance.

    Returns:
        List of DocumentMetadata.
    """
    return db.get_all_documents_metadata()


def get_chunk_action(db: DatabaseManager, chunk_id: int) -> EvidenceChunk | None:
    """Retrieve full details for a single document chunk by primary key ID.

    Args:
        db: DatabaseManager instance.
        chunk_id: Primary key ID of target chunk.

    Returns:
        EvidenceChunk or None.
    """
    return db.get_chunk_by_id(chunk_id)


def get_chunk_neighbors_action(db: DatabaseManager, chunk_id: int, window: int = 1) -> list[EvidenceChunk]:
    """Retrieve neighboring context chunks (before and after) for a given chunk ID.

    Args:
        db: DatabaseManager instance.
        chunk_id: Primary key ID of anchor chunk.
        window: Number of neighboring chunks before/after.

    Returns:
        List of EvidenceChunk.
    """
    return db.get_chunk_neighbors(chunk_id=chunk_id, window=window)


def read_doc_page_action(
    db: DatabaseManager,
    document_name: str,
    page_number: int,
    jurisdiction: str | None = None,
) -> DocumentPageInfo | None:
    """Execute page-level text retrieval for a regulatory document.

    Args:
        db: DatabaseManager instance.
        document_name: Target document filename.
        page_number: 1-indexed page number.
        jurisdiction: Optional jurisdiction filter.

    Returns:
        DocumentPageInfo or None.
    """
    return db.get_document_page(
        document_name=document_name,
        page_number=page_number,
        jurisdiction=jurisdiction,
    )


def calculate_action(expression: str) -> Decimal:
    """Safely evaluate an arithmetic expression using ast parser.

    Args:
        expression: Arithmetic expression string.

    Returns:
        Evaluated Decimal result.
    """
    return evaluate_expression(expression)


def get_financial_record_action(db: DatabaseManager, record_id: int) -> BaseStrictRecord | None:
    """Retrieve full details for a single financial record by primary key ID.

    Args:
        db: DatabaseManager instance.
        record_id: Primary key ID of target financial record.

    Returns:
        BaseStrictRecord or None.
    """
    return db.get_strict_financial_record(record_id)


def filter_financial_records_action(  # noqa: PLR0917
    db: DatabaseManager,
    asset_type: str | None = None,
    action: str | None = None,
    tax_year: int | None = None,
    isin: str | None = None,
    quantity_over: Decimal | None = None,
    quantity_less: Decimal | None = None,
    purchase_date_start: datetime | None = None,
    purchase_date_end: datetime | None = None,
    logic: str = "AND",
    account_country: str | None = None,
    limit: int = 100,
) -> list[BaseStrictRecord]:
    """Filter financial records by asset_type, action, tax_year, isin, quantity, purchase date, with AND/OR logic.

    Args:
        db: DatabaseManager instance.
        asset_type: Optional asset type string.
        action: Optional transaction action string.
        tax_year: Optional tax year integer.
        isin: Optional ISIN identifier.
        quantity_over: Optional quantity lower bound.
        quantity_less: Optional quantity upper bound.
        purchase_date_start: Optional start timestamp/date string.
        purchase_date_end: Optional end timestamp/date string.
        logic: Logical operator ('AND' or 'OR').
        account_country: Optional account country filter.
        limit: Maximum results to return.

    Returns:
        List of BaseStrictRecord.
    """
    if purchase_date_end and purchase_date_end.time() == time.min:
        purchase_date_end = datetime.combine(purchase_date_end.date(), time.max)

    return db.filter_financial_records(
        asset_type=asset_type,
        action=action,
        tax_year=tax_year,
        isin=isin,
        quantity_over=quantity_over,
        quantity_less=quantity_less,
        purchase_date_start=purchase_date_start,
        purchase_date_end=purchase_date_end,
        logic=logic,
        account_country=account_country,
        limit=limit,
    )


def get_taxpayer_profile_action(db: DatabaseManager, tax_year: int | None = None) -> list[TaxpayerProfile]:
    """Fetch taxpayer profile status (fiscal residence, domicile, residency type) by tax year or all years.

    Args:
        db: DatabaseManager instance.
        tax_year: Optional tax year integer filter.

    Returns:
        List of TaxpayerProfile.
    """
    if tax_year is not None:
        prof = db.get_taxpayer_profile(tax_year)
        if prof is not None:
            return [prof]
        return []
    return db.get_all_taxpayer_profiles()


def get_tax_income_records_action(
    db: DatabaseManager,
    tax_year: int | None = None,
    jurisdiction: str | None = None,
) -> list[StrictTaxIncomeRecord]:
    """Fetch official tax income records (EDS, CU, etc.) by tax year and jurisdiction.

    Args:
        db: DatabaseManager instance.
        tax_year: Optional tax year integer filter.
        jurisdiction: Optional jurisdiction filter ('ireland', 'italy').

    Returns:
        List of StrictTaxIncomeRecord domain models.
    """
    return db.get_tax_income_records(tax_year=tax_year, jurisdiction=jurisdiction)

