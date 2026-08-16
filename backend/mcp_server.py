import logging
import os
import sys

from fastmcp import FastMCP

from backend.db_manager import DatabaseManager, LocalDb
from backend.deliberation.models import EvidenceChunk
from backend.domain_models import DocumentMetadata, DocumentStatistics, SourceType
from backend.ingestion.chunker import LateChunker

# Standard I/O is used for JSON-RPC transport in stdio MCP servers.
# It is CRITICAL that standard output (stdout) is NEVER written to with print() or raw logs,
# as that will corrupt the transport channel.
# We redirect all standard logs and output explicitly to sys.stderr!
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


# Initialize FastMCP Server
mcp = FastMCP("Tax-Compliance-Context")

# Global instances initialized lazily on the first request
db = None
chunker = None


def get_db() -> DatabaseManager:
    global db
    if db is None:
        default_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../database/tax_data.db")
        db_path = os.environ.get("TAX_DB_PATH", default_db_path)
        db = DatabaseManager(db_config=LocalDb(db_path=db_path))
    return db


def get_chunker() -> LateChunker:
    global chunker
    if chunker is None:
        logger.info("Initializing lazily BGE-M3 model for query encoding...")
        # Automatically manages MPS/CPU device selection
        chunker = LateChunker()
    return chunker


@mcp.tool()
def list_documents() -> list[DocumentMetadata]:
    """List all tax regulatory documents, instructions, and manuals currently indexed in the vector database.

    Use this to see what source documents are available for query.
    """
    logger.info("Tool called: list_documents")
    manager = get_db()

    docs = manager.get_all_documents_metadata()
    return docs


@mcp.tool()
def query_tax_knowledge(
    query_text: str,
    limit: int = 5,
    jurisdiction: str | None = None,
    source_type: SourceType | None = None,
) -> list[EvidenceChunk]:
    """Perform a high-precision semantic search against both Italian and Irish tax guidelines and documents.

    Returns matching chunks containing context-rich text passages, original page references,
    source provenance ('regulation' or 'research'), and confidence ratings ('high', 'medium', 'low').

    Args:
        query_text: The user's query or concept (e.g., 'Offshore ETF deemed disposal exit tax rate', 'Quadro RW IVAFE').
        limit: Max number of matches to retrieve (default 5).
        jurisdiction: Optional filter for country context ('italy' or 'ireland').
        source_type: Optional filter for source provenance ('regulation' or 'research').
    """
    logger.info(
        f"Tool called: query_tax_knowledge | Query: '{query_text}' | Limit: {limit} | Jurisdiction: {jurisdiction} | SourceType: {source_type}"
    )
    manager = get_db()
    model_chunker = get_chunker()

    embedding_list = model_chunker.embed_query(query_text)

    # 2. Query vector database using KNN MATCH syntax
    return manager.semantic_search(
        embedding_list,
        limit=limit,
        jurisdiction=jurisdiction,
        source_type=source_type,
    )


@mcp.tool()
def keyword_search_knowledge(
    query_text: str,
    limit: int = 10,
    jurisdiction: str | None = None,
    source_type: SourceType | None = None,
) -> list[EvidenceChunk]:
    """Perform keyword SQL search over tax guidelines, regulations, and research documents.

    Complements semantic search for exact phrase and keyword term matching
    (e.g. specific article numbers, form codes, or statutory terms).

    Args:
        query_text: Exact search phrase or keywords.
        limit: Maximum number of matches to retrieve (default 10).
        jurisdiction: Optional filter for country context ('italy' or 'ireland').
        source_type: Optional filter for document provenance ('regulation' or 'research').
    """
    logger.info(
        f"Tool called: keyword_search_knowledge | Query: '{query_text}' | Limit: {limit} | Jurisdiction: {jurisdiction} | SourceType: {source_type}"
    )
    manager = get_db()
    return manager.keyword_search(
        query_text=query_text,
        limit=limit,
        jurisdiction=jurisdiction,
        source_type=source_type,
    )


@mcp.tool()
def get_chunk_content(chunk_id: int) -> EvidenceChunk:
    """Retrieve the full text content and precise legal reference for a specific chunk given its unique ID.

    Use this to inspect a specific passage after running a semantic search.
    """
    logger.info(f"Tool called: get_chunk_content | Chunk ID: {chunk_id}")
    manager = get_db()
    chunk = manager.get_chunk_by_id(chunk_id)

    if not chunk:
        raise ValueError(f"No chunk found with ID {chunk_id}")

    return chunk


@mcp.tool()
def get_surrounding_context(
    chunk_id: int,
    window: int = 1,
) -> list[EvidenceChunk]:
    """Retrieve neighboring chunks sequentially before and after a specific chunk ID.

    This enables dynamic context expansion, allowing agents to read the preceding and succeeding paragraphs
    of a document to capture the full legal context without splits.

    Args:
        chunk_id: The ID of the chunk to center on.
        window: Number of chunks before and after to retrieve (default 1).
    """
    logger.info(f"Tool called: get_surrounding_context | Chunk ID: {chunk_id} | Window: {window}")
    manager = get_db()
    return manager.get_chunk_neighbors(chunk_id, window=window)


@mcp.tool()
def get_document_statistics(document_sha: str) -> DocumentStatistics:
    """Retrieve size metrics for a document, showing its total sequential chunks and page counts.

    Use this to understand the scale/size of the original document reference.
    """
    logger.info(f"Tool called: get_document_statistics | SHA: {document_sha}")
    manager = get_db()

    stats = manager.get_document_statistics(document_sha)
    if not stats:
        raise ValueError(f"No statistics found for document SHA '{document_sha}'")

    return stats


if __name__ == "__main__":
    # Stdio transport runs JSON-RPC communication on stdin/stdout.
    # FastMCP starts standard stdio loops here.
    logger.info("Starting Tax Compliance Context MCP Server via stdio transport...")
    mcp.run(transport="stdio", show_banner=False)
