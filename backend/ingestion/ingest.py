import argparse
import logging
import os

from dotenv import load_dotenv

from backend.db_manager import DatabaseManager, LocalDb
from backend.domain_models import ConfidenceLevel, SourceType
from backend.ingestion.chunker import LateChunker
from backend.ingestion.helpers import (
    calculate_sha256,
    extract_jurisdiction_from_path,
    extract_research_jurisdiction_from_path,
    log_env_vars,
)
from backend.ingestion.language_detector import detect_file_language
from backend.ingestion.markdown_parser import MarkdownParser
from backend.ingestion.parser import BasePDFParser, ParsedPage, get_parser_registry

_ = load_dotenv()

PROCESSING_LOCATION: str = (
    os.environ.get("DATALAB_PROCESSING_LOCATION_REGULATIONS") or os.environ.get("DATALAB_PROCESSING_LOCATION") or "eu"
)


# Maps --parser flag value to the corresponding parser class.
PARSER_REGISTRY: dict[str, type[BasePDFParser]] = get_parser_registry()


# Maps jurisdiction string to ISO 639-1 language code used by Stanza SBD.
# Extend this mapping when new jurisdictions are added.
JURISDICTION_TO_LANGUAGE: dict[str, str] = {
    "italy": "it",
    "ireland": "en",
}

# Default batch size (number of pages) for Late Chunking to manage context window & memory.
DEFAULT_PAGE_BATCH_SIZE: int = 5


def find_page_number(char_idx: int, page_ranges: list[tuple[int, int, int]]) -> int:
    """Find the source page number for a given character index in the concatenated text.

    Args:
        char_idx: Absolute character offset in concatenated batch text.
        page_ranges: List of (start_offset, end_offset, page_number) tuples.

    Returns:
        Starting page number containing the character index.

    Raises:
        ValueError: If page_ranges is empty or char_idx cannot be mapped to a page.
    """
    if not page_ranges:
        raise ValueError("Cannot map character index: page_ranges list is empty.")

    for start, end, page_num in page_ranges:
        if start <= char_idx <= end:
            return page_num

    min_start = page_ranges[0][0]
    max_end = page_ranges[-1][1]
    raise ValueError(f"Character index {char_idx} out of bounds for page ranges [{min_start}, {max_end}].")


def _process_page_batch(
    batch_pages: list[ParsedPage],
    chunker: LateChunker,
    language: str,
) -> list[tuple[int, str, list[float], str | None]]:
    """Process a batch of parsed pages, compute late chunks, and return chunk records.

    Returns:
        List of tuples: (start_page_number, text_content, embedding, parent_text_content)
    """
    batch_text_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    current_offset = 0

    for page in batch_pages:
        content = page.combined_content
        start_idx = current_offset
        end_idx = current_offset + len(content)
        page_ranges.append((start_idx, end_idx, page.page_number))

        batch_text_parts.append(content)
        current_offset = end_idx + 1  # 1 character representing newline separator

    batch_text = "\n".join(batch_text_parts)
    chunks = chunker.compute_late_chunks(batch_text, language=language)

    records: list[tuple[int, str, list[float], str | None]] = []
    for chunk in chunks:
        # Note: page_num represents the starting page of the chunk
        page_num = find_page_number(chunk.start_char_idx, page_ranges)
        parent_text = chunk.parent_text if chunk.parent_text else None
        records.append((page_num, chunk.text_content, chunk.embedding, parent_text))

    return records


def ingest_document(  # noqa: PLR0917
    file_path: str,
    db: DatabaseManager,
    chunker: LateChunker,
    jurisdiction: str,
    force: bool = False,
    force_ocr: bool = False,
    parser: str = "pdfplumber",
    batch_size: int = DEFAULT_PAGE_BATCH_SIZE,
) -> None:
    """Parse, chunk, and index a single PDF document in the SQLite database.

    Args:
        file_path: Path to PDF file.
        db: DatabaseManager instance.
        chunker: LateChunker instance.
        jurisdiction: Jurisdiction string ('italy' or 'ireland').
        force: If True, force re-chunking and rebuilding database records.
        force_ocr: If True, bypass local OCR cache and force raw parsing/API call.
        parser: PDF parser key in PARSER_REGISTRY.
        batch_size: Number of pages to process per batch.
    """
    document_name = os.path.basename(file_path)
    document_sha = calculate_sha256(file_path)
    language = JURISDICTION_TO_LANGUAGE.get(jurisdiction, "en")

    if db.is_document_ingested(document_sha) and not force:
        print(f"Skipping document '{document_name}' (already ingested).")
        return

    print(f"\nIngesting: {document_name} ({jurisdiction.upper()}, lang={language})")

    # 1. Parse PDF page-by-page using the selected parser interface
    parser_class = PARSER_REGISTRY[parser]
    pages: list[ParsedPage] = parser_class.parse_pdf(file_path, force_parsing=force_ocr)

    if not pages:
        print(f"Warning: No pages parsed from {file_path}")
        return

    print(f"Running Late Chunking via BGE-M3 (total pages: {len(pages)})...")

    # 2. Process pages in batches to compute embeddings before DB mutation
    processed_chunks: list[tuple[int, str, list[float], str | None]] = []
    total_pages = len(pages)

    for i in range(0, total_pages, batch_size):
        batch_pages = pages[i : i + batch_size]
        print(f"  Processing pages {i + 1} to {min(i + batch_size, total_pages)} of {total_pages}...")
        batch_records = _process_page_batch(batch_pages, chunker, language)
        processed_chunks.extend(batch_records)

    # 3. Only delete existing DB document after successful parsing and chunking
    if db.is_document_ingested(document_sha) and force:
        print(f"Document '{document_name}' already ingested. Rebuilding (--force active)...")
        db.delete_document(document_sha)

    # 4. Insert chunks into database
    # doc_chunk_index: Zero-based sequential index tracking chunk order within the document
    for doc_chunk_index, (page_num, text_content, embedding, parent_text) in enumerate(processed_chunks):
        _ = db.insert_chunk(
            document_name=document_name,
            jurisdiction=jurisdiction,
            page_number=page_num,
            text_content=text_content,
            chunk_index=doc_chunk_index,
            embedding=embedding,
            document_sha=document_sha,
            parent_text_content=parent_text,
            source_type=SourceType.REGULATION,
            confidence_level=ConfidenceLevel.HIGH,
        )

    print(f"Successfully ingested '{document_name}' into vector database ({len(processed_chunks)} total chunks).")


def ingest_research_document(  # noqa: PLR0917
    file_path: str,
    db: DatabaseManager,
    chunker: LateChunker,
    confidence: ConfidenceLevel,
    *,
    jurisdiction: str | None = None,
    force: bool = False,
) -> None:
    """Parse, chunk, and index a single Markdown research document in the database.

    Args:
        file_path: Path to markdown file.
        db: DatabaseManager instance.
        chunker: LateChunker instance.
        jurisdiction: Optional jurisdiction string ('italy', 'ireland', etc.).
        confidence: Confidence level ('high', 'medium', 'low'). Defaults to 'medium'.
        force: If True, force re-chunking and rebuilding database records.
    """
    document_name = os.path.basename(file_path)
    document_sha = calculate_sha256(file_path)
    language = detect_file_language(file_path)

    if db.is_document_ingested(document_sha) and not force:
        print(f"Skipping research document '{document_name}' (already ingested).")
        return

    jur_label = jurisdiction.upper() if jurisdiction else "None"
    print(f"\nIngesting Research: {document_name} ({jur_label}, confidence={confidence}, lang={language})")

    page: ParsedPage | None = MarkdownParser.parse_markdown(file_path)
    if not page:
        print(f"Warning: No sections parsed from {file_path}")
        return

    print("Running Late Chunking via BGE-M3.")

    processed_chunks: list[tuple[int, str, list[float], str | None]] = []

    batch_records = _process_page_batch([page], chunker, language)
    processed_chunks.extend(batch_records)

    if db.is_document_ingested(document_sha) and force:
        print(f"Research document '{document_name}' already ingested. Rebuilding (--force active)...")
        db.delete_document(document_sha)

    for doc_chunk_index, (page_num, text_content, embedding, parent_text) in enumerate(processed_chunks):
        _ = db.insert_chunk(
            document_name=document_name,
            jurisdiction=jurisdiction,
            page_number=page_num,
            text_content=text_content,
            chunk_index=doc_chunk_index,
            embedding=embedding,
            document_sha=document_sha,
            parent_text_content=parent_text,
            source_type=SourceType.RESEARCH,
            confidence_level=ConfidenceLevel.MEDIUM,
        )

    print(
        f"Successfully ingested research '{document_name}' into vector database ({len(processed_chunks)} total chunks)."
    )


def main():

    log_env_vars(logging.getLogger(__name__))

    arg_parser = argparse.ArgumentParser(
        description="Tax Document Batch Ingestion Subsystem (Late Chunking + sqlite-vec)"
    )
    _ = arg_parser.add_argument(
        "--db",
        type=str,
        default="database/tax_data.db",
        help="Path to SQLite database file",
    )
    _ = arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Force both re-chunking and raw OCR parsing from scratch",
    )
    _ = arg_parser.add_argument(
        "--force-chunking",
        "--force_chunking",
        dest="force_chunking",
        action="store_true",
        help="Force re-chunking and rebuilding database records (uses cached OCR if available)",
    )
    _ = arg_parser.add_argument(
        "--force-ocr",
        "--force_ocr",
        dest="force_ocr",
        action="store_true",
        help="Force re-running OCR parser/API from scratch (bypasses local OCR cache)",
    )
    _ = arg_parser.add_argument(
        "--file",
        type=str,
        help="Ingest a specific PDF or Markdown file instead of checking default directories",
    )
    _ = arg_parser.add_argument(
        "--parser",
        type=str,
        default="chandra",
        choices=list(PARSER_REGISTRY.keys()),
        help="PDF extraction backend to use (default: chandra)",
    )
    _ = arg_parser.add_argument(
        "--research-dir",
        type=str,
        default="data/research",
        help="Path to research markdown files directory (default: data/research)",
    )
    _ = arg_parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["regulations", "research", "all"],
        help="Ingestion mode: regulations (PDFs), research (Markdown), or all (default: all)",
    )
    args = arg_parser.parse_args()

    effective_force = args.force or args.force_chunking or args.force_ocr
    effective_force_ocr = args.force or args.force_ocr

    # Initialize DB Manager
    db = DatabaseManager(db_config=LocalDb(db_path=args.db))

    # Initialize Late Chunker (BGE-M3)
    chunker = LateChunker()

    try:
        # Single file ingestion mode
        if args.file:
            if not os.path.exists(args.file):
                print(f"Error: Specified file does not exist: {args.file}")
                return

            if args.file.lower().endswith(".pdf"):
                jurisdiction = extract_jurisdiction_from_path(args.file)
                ingest_document(
                    args.file,
                    db,
                    chunker,
                    jurisdiction,
                    force=effective_force,
                    force_ocr=effective_force_ocr,
                    parser=args.parser,
                )
                return
            elif args.file.lower().endswith(".md"):
                jurisdiction = extract_research_jurisdiction_from_path(args.file)
                ingest_research_document(
                    args.file,
                    db,
                    chunker,
                    jurisdiction=jurisdiction,
                    confidence=ConfidenceLevel.MEDIUM,
                    force=effective_force,
                )
                return
            else:
                print("Error: Ingestion only supports .pdf and .md files.")
                return

        # Batch directory scanning mode: Regulations
        if args.mode in ("regulations", "all"):
            raw_sources_dir = "data/raw_sources/regulations"
            if os.path.exists(raw_sources_dir):
                for root, _dirs, files in os.walk(raw_sources_dir):
                    for file_name in files:
                        if str(file_name).lower().endswith(".pdf"):
                            pdf_path: str = os.path.join(str(root), str(file_name))
                            jurisdiction = extract_jurisdiction_from_path(pdf_path)
                            ingest_document(
                                pdf_path,
                                db,
                                chunker,
                                jurisdiction,
                                force=effective_force,
                                force_ocr=effective_force_ocr,
                                parser=args.parser,
                            )
            else:
                print(f"Warning: Regulations directory '{raw_sources_dir}' not found.")

        # Batch directory scanning mode: Research
        if args.mode in ("research", "all"):
            research_dir = str(args.research_dir)
            if os.path.exists(research_dir):
                for root, _dirs, files in os.walk(research_dir):
                    for file_name in files:
                        if str(file_name).lower().endswith(".md"):
                            md_path: str = os.path.join(str(root), str(file_name))
                            ingest_research_document(
                                md_path,
                                db,
                                chunker,
                                jurisdiction=None,
                                confidence=ConfidenceLevel.MEDIUM,
                                force=effective_force,
                            )
            else:
                print(f"Warning: Research directory '{research_dir}' not found.")

        print("\nBatch Ingestion execution completed successfully!")

    finally:
        db.close()


if __name__ == "__main__":
    main()
