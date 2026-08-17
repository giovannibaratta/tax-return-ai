"""Database Manager for Tax Return AI.

Architecture (Dual-Database Design):
-----------------------------------
1. Relational Database (`tax_data.db`):
   - Stores structured financial records, tax document metadata, taxpayer profiles, etc.
   - Schema DDL and migrations are managed strictly via Liquibase CLI (db-migrations/).
   - Uses standard SQLite without C extensions.

2. Vector Database (`tax_vectors.db`):
   - Stores `vss_tax_chunks` virtual table (`sqlite-vec` `vec0`) for RAG embeddings.
   - Managed directly by Python using C extension loading (`sqlite_vec.load`).

Why Two Database Files?
-----------------------
Liquibase Java CLI uses standard `sqlite-jdbc` without native C extension loading capabilities.
If `vec0` virtual tables are present in `tax_data.db`, Liquibase schema inspection crashes
with `no such module: vec0`. Isolating vector storage into `tax_vectors.db` keeps `tax_data.db`
100% standard SQLite, eliminating Liquibase migration failures on existing databases.
"""

import hashlib
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from warnings import deprecated

import sqlite_vec
from sqlalchemy import PoolProxiedConnection, and_, event, func, or_, text
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.sql.expression import ColumnElement
from sqlmodel import Session, SQLModel, col, create_engine, select

from backend.db_models import (
    AssetMerger,
    FinancialRecord,
    IngestedSourceDocument,
    StagedFinancialRecord,
    TaxDocumentMetadata,
    TaxIncomeRecord,
)
from backend.deliberation.models import EvidenceChunk
from backend.domain_models import (
    BaseStrictRecord,
    ConfidenceLevel,
    DocumentMetadata,
    DocumentPageInfo,
    DocumentStatistics,
    IngestionDocumentSummary,
    IngestionStatus,
    SourceType,
    StrictTaxIncomeRecord,
    TradeRecord,
)
from src.jurisdiction.ireland.cgt_models import AssetTaxClassification, RemittanceEvent, TaxpayerProfile

DEFAULT_DB_PATH = "database/tax_data.db"
DEFAULT_VECTOR_DB_PATH = "database/tax_vectors.db"


@dataclass(frozen=True)
class MemoryDb:
    """In-memory SQLite database configuration."""

    @property
    def db_path(self) -> str:
        return ":memory:"

    @property
    def vector_db_path(self) -> str:
        return ":memory:"


@dataclass(frozen=True)
class LocalDb:
    """Local SQLite file database configuration."""

    db_path: str | Path = DEFAULT_DB_PATH
    vector_db_path: str | Path = DEFAULT_VECTOR_DB_PATH


def _run_migrations(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Run Liquibase YAML migrations to ensure the database schema is up-to-date.

    Args:
        db_path: Path to the SQLite database file passed to Liquibase.

    Raises:
        RuntimeError: If the Liquibase executable is not found or migration fails.
    """
    try:
        _ = subprocess.run(
            [
                "liquibase",
                "--changelog-file=db-migrations/root-changelog.yaml",
                f"--url=jdbc:sqlite:{db_path}",
                "update",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print("  * Liquibase migrations completed successfully.")
    except FileNotFoundError as e:
        print("  * ERROR: Liquibase executable not found. Cannot run migrations.")
        raise RuntimeError(
            "Liquibase executable not found. Please install it (e.g., brew install liquibase) to run database migrations."
        ) from e
    except subprocess.CalledProcessError as e:
        print(f"  * ERROR: Liquibase migration failed with exit code {e.returncode}")
        print(f"  * Liquibase stdout: {e.stdout}")
        print(f"  * Liquibase stderr: {e.stderr}")
        raise RuntimeError("Database migration failed") from e


class DatabaseManager:
    """Manages SQLite connection, schema migrations, and vector/relational queries.

    Dual-Database Architecture:
    - Relational engine (`tax_data.db`): Liquibase-managed standard SQLite schema.
    - Vector engine (`tax_vectors.db`): `sqlite-vec` managed embedding store (`vss_tax_chunks`).
    """

    def __init__(
        self,
        db_config: MemoryDb | LocalDb = LocalDb(),
        auto_migrate: bool = True,
    ):
        self.config: MemoryDb | LocalDb = db_config
        self.db_path = str(self.config.db_path)
        self.vector_db_path = str(self.config.vector_db_path)

        # Run Liquibase migrations automatically for DEFAULT_DB_PATH
        if auto_migrate and isinstance(self.config, LocalDb):
            _run_migrations(self.db_path)

        # Ensure database directories exist for local file databases
        if isinstance(self.config, LocalDb):
            for path in (self.db_path, self.vector_db_path):
                db_dir = os.path.dirname(path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)

        # Main relational SQLModel / SQLAlchemy engine (Liquibase-managed)
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})

        # Dedicated vector engine with sqlite-vec extension loaded
        self.vector_engine = create_engine(
            f"sqlite:///{self.vector_db_path}", connect_args={"check_same_thread": False}
        )

        @event.listens_for(self.vector_engine, "connect")
        def load_vec_extension(
            dbapi_connection: sqlite3.Connection, connection_record: ConnectionPoolEntry
        ) -> None:
            dbapi_connection.enable_load_extension(True)
            try:
                sqlite_vec.load(dbapi_connection)
            except Exception as e:
                print(f"Error loading sqlite-vec extension on vector engine: {e}")
                raise RuntimeError("Failed to load sqlite-vec extension on database connection.") from e
            finally:
                dbapi_connection.enable_load_extension(False)

        self._init_vector_db()

    def _init_vector_db(self) -> None:
        """Initialize SQLModel tables and the sqlite-vec virtual table (vss_tax_chunks)."""
        SQLModel.metadata.create_all(self.engine)

        with self.vector_engine.begin() as v_conn:
            v_conn.execute(
                text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_tax_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT[1024]
                );
            """)
            )

    def insert_chunk(  # noqa: PLR0917
        self,
        document_name: str,
        jurisdiction: str | None,
        page_number: int,
        text_content: str,
        chunk_index: int,
        embedding: list[float],
        document_sha: str,
        source_type: SourceType,
        confidence_level: ConfidenceLevel,
        *,
        parent_chunk_id: int | None = None,
        parent_text_content: str | None = None,
    ) -> int:
        """Insert a chunk's metadata into relational DB and vector embedding into vector store.

        Args:
            document_name: Base filename of the source PDF or markdown document.
            jurisdiction: Jurisdiction string (e.g. 'italy', 'ireland', or None for research).
            page_number: Page number the chunk was extracted from.
            text_content: Child chunk text (small, precise retrieval unit).
            chunk_index: Zero-based chunk position within the document.
            embedding: Normalized 1024-dimensional BGE-M3 float vector.
            document_sha: SHA-256 of the source document file.
            parent_chunk_id: Primary key of the parent chunk row, or ``None``
                for rows that are themselves parents.
            parent_text_content: Larger parent context text (~PARENT_MAX_CHARS)
                stored denormalized for fast retrieval without JOIN.
            source_type: Source category ('regulation' or 'research').
            confidence_level: Confidence level ('high', 'medium', or 'low').

        Returns:
            The integer primary key assigned to the inserted chunk row.
        """
        # Generate stable chunk ID from document_sha and chunk_index
        h = hashlib.sha256(f"{document_sha}_{chunk_index}".encode()).digest()
        chunk_id = abs(int.from_bytes(h[:8], byteorder="big", signed=True))

        doc = TaxDocumentMetadata(
            id=chunk_id,
            document_name=document_name,
            document_sha=document_sha,
            jurisdiction=jurisdiction,
            source_type=source_type.value,
            confidence_level=confidence_level.value,
            page_number=page_number,
            text_content=text_content,
            chunk_index=chunk_index,
            parent_chunk_id=parent_chunk_id,
            parent_text_content=parent_text_content,
        )

        with Session(self.engine) as session:
            session.add(doc)
            session.commit()

        # Serialize the 1024-dimensional vector to raw float32 BLOB for sqlite-vec
        serialized_vector = sqlite_vec.serialize_float32(embedding)

        # Insert vector embedding into the dedicated vector database engine
        with Session(self.vector_engine) as v_session:
            _ = v_session.connection().execute(
                text("""
                INSERT INTO vss_tax_chunks (chunk_id, embedding)
                VALUES (:chunk_id, :embedding)
            """),
                {"chunk_id": chunk_id, "embedding": serialized_vector},
            )
            v_session.commit()

        return chunk_id

    def is_document_ingested(self, document_sha: str) -> bool:
        """Check if a document has already been ingested by its SHA-256 hash."""
        with Session(self.engine) as session:
            statement = select(TaxDocumentMetadata).where(TaxDocumentMetadata.document_sha == document_sha).limit(1)
            result = session.exec(statement).first()
            return result is not None

    def keyword_search(
        self,
        query_text: str,
        *,
        limit: int = 50,
        jurisdiction: str | None = None,
        source_type: SourceType | None = None,
    ) -> list[EvidenceChunk]:
        """Perform keyword SQL LIKE search over tax chunk text content to complement vector search.

        Args:
            query_text: Natural language query string.
            limit: Maximum candidate chunks to retrieve.
            jurisdiction: Optional jurisdiction filter ('italy' or 'ireland').
            source_type: Optional source type filter ('regulation' or 'research').

        Returns:
            List of EvidenceChunk objects matching keyword query conditions.
        """
        if not query_text or not query_text.strip():
            return []

        clean_q = query_text.strip().lower()
        words = [w.strip() for w in clean_q.split() if len(w.strip()) > 1]
        if not words:
            return []

        norm_jurisdiction = jurisdiction.strip().lower() if jurisdiction else None

        # Construct 2-word (bigrams) and 3-word (trigrams) contiguous phrases from query.
        # Matching multi-word phrases via SQL contains() yields significantly higher precision
        # for technical/tax queries than matching isolated unigram words.
        phrases: list[str] = []
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i + 1]}")
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")

        with Session(self.engine) as session:
            stmt = select(TaxDocumentMetadata)
            if norm_jurisdiction:
                stmt = stmt.where(func.lower(TaxDocumentMetadata.jurisdiction) == norm_jurisdiction)
            if source_type:
                stmt = stmt.where(TaxDocumentMetadata.source_type == source_type)

            conditions: list[ColumnElement[bool]] = []
            if phrases:
                conditions.extend([func.lower(TaxDocumentMetadata.text_content).contains(p) for p in phrases])

            if conditions:
                stmt = stmt.where(or_(*conditions)).limit(limit)
            else:
                stmt = stmt.limit(limit)

            records = session.exec(stmt).all()
            return [
                EvidenceChunk(
                    id=rec.id,
                    document_name=rec.document_name,
                    jurisdiction=rec.jurisdiction,
                    page_number=rec.page_number,
                    chunk_index=rec.chunk_index,
                    text_content=rec.text_content,
                    # Note: Keyword search returns exact textual hits without KNN vector distance.
                    # distance is assigned 0.0 as a baseline ranking value in EvidenceChunk DTOs.
                    distance=0.0,
                    # Fallback to child chunk text when parent_text_content was not denormalized during ingestion.
                    parent_text_content=rec.parent_text_content or rec.text_content,
                    source_type=SourceType(rec.source_type),
                    confidence_level=ConfidenceLevel(rec.confidence_level),
                )
                for rec in records
            ]

    def delete_document(self, document_sha: str):
        """Remove a document's metadata and associated vector embeddings using SQLModel."""
        with Session(self.engine) as session:
            # Find all associated chunk IDs
            statement = select(TaxDocumentMetadata.id).where(TaxDocumentMetadata.document_sha == document_sha)
            chunk_ids = session.exec(statement).all()

            if not chunk_ids:
                return

            # Delete from metadata table
            delete_statement = select(TaxDocumentMetadata).where(TaxDocumentMetadata.document_sha == document_sha)
            records = session.exec(delete_statement).all()
            for record in records:
                session.delete(record)
            session.commit()

        # Delete from vector virtual table
        with Session(self.vector_engine) as v_session:
            v_session.connection().execute(
                text("DELETE FROM vss_tax_chunks WHERE chunk_id = :cid"), [{"cid": cid} for cid in chunk_ids]
            )
            v_session.commit()

        print(f"Removed document with SHA '{document_sha}' and its {len(chunk_ids)} chunks from the database.")

    def delete_all_chunks(self) -> int:
        """Remove all document metadata chunks and vector embeddings from the database.

        Note:
            This clears all regulatory chunks (`tax_document_metadata` and `vss_tax_chunks`)
            without clearing cached OCR/parsing artifacts or financial records.

        Returns:
            The total number of chunk metadata records deleted.
        """
        with Session(self.engine) as session:
            statement = select(TaxDocumentMetadata)
            records = list(session.exec(statement).all())
            count = len(records)

            for record in records:
                session.delete(record)
            session.commit()

        with Session(self.vector_engine) as v_session:
            v_session.connection().execute(text("DELETE FROM vss_tax_chunks"))
            v_session.commit()

        print(f"Removed all {count} chunks and vector embeddings from the database.")
        return count

    @staticmethod
    def normalize_jurisdiction(jurisdiction: str | None) -> str | None:
        """Normalize jurisdiction string aliases (e.g. 'IE', 'Irish' -> 'ireland'; 'IT', 'Italian' -> 'italy')."""
        if not jurisdiction:
            return None
        s = jurisdiction.strip().lower()
        if s in ("ie", "ireland", "irish"):
            return "ireland"
        if s in ("it", "italy", "italian", "italia"):
            return "italy"
        return s

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        jurisdiction: str | None = None,
        source_type: SourceType | None = None,
    ) -> list[EvidenceChunk]:
        """Perform KNN vector search using sqlite-vec and return typed EvidenceChunk results.

        Args:
            query_embedding: A normalized 1024-dimensional float vector representing the query.
            limit: Maximum number of results to return.
            jurisdiction: Optional jurisdiction filter (e.g. 'italy', 'ireland', 'IE', 'IT').
                When provided, jurisdiction is normalized and KNN candidate pool is over-fetched
                (limit * 10) to ensure accurate filtering.
            source_type: Optional source type filter (e.g. 'regulation' or 'research').

        Returns:
            A list of EvidenceChunk objects ordered by ascending cosine distance.
        """
        serialized_query = sqlite_vec.serialize_float32(query_embedding)
        norm_jurisdiction = DatabaseManager.normalize_jurisdiction(jurisdiction)
        # Vector table `vss_tax_chunks` in tax_vectors.db stores vector embeddings without relational metadata.
        # When a jurisdiction or source_type filter is active, candidate KNN results are over-fetched
        # (limit * 10, min 100) from the vector engine to guarantee sufficient candidates remain after relational filtering.
        fetch_k = max(limit * 10, 100) if (norm_jurisdiction or source_type) else limit

        with Session(self.vector_engine) as v_session:
            result = v_session.connection().execute(
                text("""
                    SELECT chunk_id, distance
                    FROM vss_tax_chunks
                    WHERE embedding MATCH :query AND k = :fetch_k
                """),
                {"query": serialized_query, "fetch_k": fetch_k},
            )
            vec_rows = result.fetchall()

        if not vec_rows:
            return []

        dist_map = {row[0]: row[1] for row in vec_rows}
        chunk_ids = list(dist_map.keys())

        with Session(self.engine) as session:
            stmt = select(TaxDocumentMetadata).where(col(TaxDocumentMetadata.id).in_(chunk_ids))
            if norm_jurisdiction:
                stmt = stmt.where(func.lower(TaxDocumentMetadata.jurisdiction) == norm_jurisdiction)
            if source_type:
                stmt = stmt.where(TaxDocumentMetadata.source_type == source_type)
            records = session.exec(stmt).all()

        evidence_chunks = [
            EvidenceChunk(
                distance=dist_map.get(rec.id, 0.0),
                document_name=rec.document_name,
                jurisdiction=rec.jurisdiction,
                page_number=rec.page_number,
                text_content=rec.text_content,
                chunk_index=rec.chunk_index,
                id=rec.id,
                parent_chunk_id=rec.parent_chunk_id,
                parent_text_content=rec.parent_text_content or rec.text_content,
                source_type=SourceType(rec.source_type),
                confidence_level=ConfidenceLevel(rec.confidence_level),
            )
            for rec in records
        ]

        evidence_chunks.sort(key=lambda x: x.distance if x.distance is not None else 0.0)
        return evidence_chunks[:limit]

    def get_chunk_by_id(self, chunk_id: int) -> EvidenceChunk | None:
        """Retrieve a single regulatory document chunk by its primary key.

        Args:
            chunk_id: The integer primary key of the chunk in ``tax_document_metadata``.

        Returns:
            An ``EvidenceChunk`` if found, or ``None`` if the ID does not exist.
        """
        with Session(self.engine) as session:
            record = session.get(TaxDocumentMetadata, chunk_id)
            if record is None:
                return None
            return EvidenceChunk(
                id=record.id,
                document_name=record.document_name,
                jurisdiction=record.jurisdiction,
                page_number=record.page_number,
                text_content=record.text_content,
                chunk_index=record.chunk_index,
                parent_chunk_id=record.parent_chunk_id,
                parent_text_content=record.parent_text_content,
                source_type=SourceType(record.source_type),
                confidence_level=ConfidenceLevel(record.confidence_level),
            )

    def get_chunks_by_parent(self, parent_chunk_id: int) -> list[EvidenceChunk]:
        """Return all child chunks that share the given parent chunk ID.

        Useful for context-window expansion: given a retrieved child, fetch its
        siblings to give the LLM the full parent context.

        Args:
            parent_chunk_id: The primary key of the parent chunk row.

        Returns:
            List of ``EvidenceChunk`` objects ordered by ``chunk_index``.
        """
        with Session(self.engine) as session:
            statement = (
                select(TaxDocumentMetadata)
                .where(TaxDocumentMetadata.parent_chunk_id == parent_chunk_id)
                .order_by(col(TaxDocumentMetadata.chunk_index))
            )
            records = list(session.exec(statement).all())

        return [
            EvidenceChunk(
                id=r.id,
                document_name=r.document_name,
                jurisdiction=r.jurisdiction,
                page_number=r.page_number,
                text_content=r.text_content,
                chunk_index=r.chunk_index,
                parent_chunk_id=r.parent_chunk_id,
                parent_text_content=r.parent_text_content,
                source_type=SourceType(r.source_type),
                confidence_level=ConfidenceLevel(r.confidence_level),
            )
            for r in records
        ]

    def get_chunk_by_document_and_index(self, document_name: str, chunk_index: int) -> EvidenceChunk | None:
        """Retrieve a specific chunk by document name and zero-based chunk index.

        Args:
            document_name: Name of the source document.
            chunk_index: Zero-based index of the chunk in the document.

        Returns:
            An EvidenceChunk if found, or None.
        """
        with Session(self.engine) as session:
            statement = select(TaxDocumentMetadata).where(
                TaxDocumentMetadata.document_name == document_name,
                TaxDocumentMetadata.chunk_index == chunk_index,
            )
            record = session.exec(statement).first()
            if record is None:
                return None
            return EvidenceChunk(
                id=record.id,
                document_name=record.document_name,
                jurisdiction=record.jurisdiction,
                page_number=record.page_number,
                text_content=record.text_content,
                chunk_index=record.chunk_index,
                parent_chunk_id=record.parent_chunk_id,
                parent_text_content=record.parent_text_content,
                source_type=SourceType(record.source_type),
                confidence_level=ConfidenceLevel(record.confidence_level),
            )

    def get_chunk_neighbors(self, chunk_id: int, window: int = 1) -> list[EvidenceChunk]:
        """Return surrounding context chunks (before and after) for a given chunk ID in the same document.

        Args:
            chunk_id: Primary key ID of the anchor chunk.
            window: Number of chunks before and after to retrieve (default: 1).

        Returns:
            List of EvidenceChunk objects ordered by chunk_index.
        """
        target = self.get_chunk_by_id(chunk_id)
        if target is None:
            return []

        min_index = max(0, target.chunk_index - window)
        max_index = target.chunk_index + window

        with Session(self.engine) as session:
            statement = (
                select(TaxDocumentMetadata)
                .where(
                    TaxDocumentMetadata.document_name == target.document_name,
                    TaxDocumentMetadata.chunk_index >= min_index,
                    TaxDocumentMetadata.chunk_index <= max_index,
                )
                .order_by(col(TaxDocumentMetadata.chunk_index))
            )
            records = list(session.exec(statement).all())

        return [
            EvidenceChunk(
                id=r.id,
                document_name=r.document_name,
                jurisdiction=r.jurisdiction,
                page_number=r.page_number,
                text_content=r.text_content,
                chunk_index=r.chunk_index,
                parent_chunk_id=r.parent_chunk_id,
                parent_text_content=r.parent_text_content,
                source_type=SourceType(r.source_type),
                confidence_level=ConfidenceLevel(r.confidence_level),
            )
            for r in records
        ]

    def get_document_page(
        self,
        document_name: str,
        page_number: int,
        jurisdiction: str | None = None,
    ) -> DocumentPageInfo | None:
        """Retrieve full page content by concatenating all chunks on page_number for document_name.

        Args:
            document_name: Name of the target document (e.g. 'manual.pdf').
            page_number: 1-indexed page number.
            jurisdiction: Optional jurisdiction filter ('ireland', 'italy', etc.).

        Returns:
            DocumentPageInfo containing page text, document name, page number, total pages, or None if page not found.
        """
        norm_jurisdiction = DatabaseManager.normalize_jurisdiction(jurisdiction)
        with Session(self.engine) as session:
            query = select(TaxDocumentMetadata).where(
                func.lower(TaxDocumentMetadata.document_name) == document_name.lower(),
                TaxDocumentMetadata.page_number == page_number,
            )
            if norm_jurisdiction:
                query = query.where(func.lower(TaxDocumentMetadata.jurisdiction) == norm_jurisdiction)

            query = query.order_by(col(TaxDocumentMetadata.chunk_index))
            chunks = list(session.exec(query).all())

            if not chunks:
                return None

            # Get max page number for total_pages count
            max_page_query = select(func.max(TaxDocumentMetadata.page_number)).where(
                func.lower(TaxDocumentMetadata.document_name) == document_name.lower()
            )
            total_pages_val = session.exec(max_page_query).one()
            total_pages = int(total_pages_val) if total_pages_val is not None else page_number

            full_page_text = "\n\n".join(c.text_content for c in chunks)
            detected_jurisdiction = chunks[0].jurisdiction

            return DocumentPageInfo(
                document_name=chunks[0].document_name,
                jurisdiction=detected_jurisdiction,
                page_number=page_number,
                total_pages=total_pages,
                text_content=full_page_text,
                chunk_count=len(chunks),
                has_previous_page=page_number > 1,
                has_next_page=page_number < total_pages,
            )

    def get_all_processed_document_pages(self) -> list[DocumentPageInfo]:
        """Fetch all indexed document pages across all jurisdictions and documents from database.

        Returns:
            List of DocumentPageInfo domain objects containing page text and metadata.
        """
        with Session(self.engine) as session:
            stmt = select(
                TaxDocumentMetadata.jurisdiction,
                TaxDocumentMetadata.document_name,
                TaxDocumentMetadata.page_number,
            ).distinct()
            pages = session.exec(stmt).all()

            page_list: list[DocumentPageInfo] = []
            for jur, doc_name, p_num in pages:
                page_data = self.get_document_page(document_name=doc_name, page_number=p_num, jurisdiction=jur)
                if page_data:
                    page_list.append(page_data)

            return page_list

    # --- Regulation document query methods ---

    def get_ingested_documents(self) -> list[IngestionDocumentSummary]:
        """Return a summary row for every distinct ingested regulation or research document.

        Returns:
            List of IngestionDocumentSummary objects ordered by document_name.
        """
        sql = text("""
            SELECT document_name, document_sha, jurisdiction, COUNT(*) AS chunk_count, source_type
            FROM tax_document_metadata
            GROUP BY document_sha, document_name, jurisdiction, source_type
            ORDER BY document_name
        """)
        with Session(self.engine) as session:
            rows = session.connection().execute(sql).fetchall()
        return [
            IngestionDocumentSummary(
                document_name=row[0],
                document_sha=row[1],
                jurisdiction=row[2],
                chunk_count=row[3],
                source_type=row[4],
            )
            for row in rows
        ]

    def get_all_documents_metadata(self) -> list[DocumentMetadata]:
        """Retrieve aggregated metadata for all documents."""
        with Session(self.engine) as session:
            sql = text("""
                SELECT document_name, jurisdiction, COUNT(*) as total_chunks, MIN(page_number) as start_page, MAX(page_number) as end_page, document_sha, source_type
                FROM tax_document_metadata
                GROUP BY document_sha, document_name, jurisdiction, source_type
                ORDER BY jurisdiction ASC, document_name ASC
            """)
            rows = session.connection().execute(sql).fetchall()

        return [
            DocumentMetadata(
                document_name=row[0],
                jurisdiction=row[1],
                total_chunks=row[2],
                page_range=f"{row[3]}-{row[4]}",
                document_sha=row[5],
                source_type=row[6],
            )
            for row in rows
        ]

    def get_document_statistics(self, document_sha: str) -> DocumentStatistics | None:
        """Retrieve size metrics for a document using its SHA."""
        with Session(self.engine) as session:
            sql = text("""
                SELECT document_name, jurisdiction, COUNT(*), MIN(page_number), MAX(page_number)
                FROM tax_document_metadata
                WHERE document_sha = :sha
                GROUP BY document_sha, document_name, jurisdiction
            """)
            row = session.connection().execute(sql, {"sha": document_sha}).fetchone()

        if not row or row[2] == 0:
            return None

        return DocumentStatistics(
            document_name=row[0],
            jurisdiction=row[1],
            total_chunks=row[2],
            start_page=row[3],
            end_page=row[4],
            total_pages=row[4] - row[3] + 1,
        )

    def get_chunks_for_document(self, document_sha: str) -> list[TaxDocumentMetadata]:
        """Return all chunks belonging to a specific document, ordered by chunk index.

        Args:
            document_sha: SHA-256 hash of the source document.

        Returns:
            List of TaxDocumentMetadata rows ordered by chunk_index.
        """
        with Session(self.engine) as session:
            statement = (
                select(TaxDocumentMetadata)
                .where(TaxDocumentMetadata.document_sha == document_sha)
                .order_by(col(TaxDocumentMetadata.chunk_index))
            )
            return list(session.exec(statement).all())

    def is_source_document_ingested(self, file_sha: str) -> bool:
        """Check if a transaction source PDF document has already been ingested successfully.

        Args:
            file_sha: SHA-256 hash of the source document file.

        Returns:
            True if document exists in ingested_source_documents with status='SUCCESS', else False.
        """
        with Session(self.engine) as session:
            statement = select(IngestedSourceDocument).where(
                IngestedSourceDocument.file_sha == file_sha, IngestedSourceDocument.status == "SUCCESS"
            )
            return session.exec(statement).first() is not None

    def get_ingested_source_document(self, file_sha: str) -> IngestedSourceDocument | None:
        """Get the ingestion tracking record for a source document SHA."""
        with Session(self.engine) as session:
            statement = select(IngestedSourceDocument).where(IngestedSourceDocument.file_sha == file_sha)
            return session.exec(statement).first()

    def upsert_ingested_source_document(  # noqa: PLR0917
        self,
        file_sha: str,
        file_name: str,
        provider: str,
        account_country: str,
        status: IngestionStatus = IngestionStatus.SUCCESS,
        transaction_count: int = 0,
        error_message: str | None = None,
    ) -> IngestedSourceDocument:
        """Record or update ingestion status for a source PDF document.

        Args:
            file_sha: SHA-256 hash of the source document file.
            file_name: Base filename of the source document.
            provider: Financial service provider name.
            account_country: Account country identifier ('italy', 'ireland').
            status: IngestionStatus enum ('SUCCESS' or 'FAILED').
            transaction_count: Number of parsed financial transactions.
            error_message: Optional error message if status is FAILED.

        Returns:
            The created or updated IngestedSourceDocument entity.
        """
        status_str = status.value
        with Session(self.engine) as session:
            statement = select(IngestedSourceDocument).where(IngestedSourceDocument.file_sha == file_sha)
            existing = session.exec(statement).first()

            if existing:
                existing.file_name = file_name
                existing.provider = provider
                existing.account_country = account_country
                existing.status = status_str
                existing.transaction_count = transaction_count
                existing.error_message = error_message
                existing.processed_at = datetime.now(timezone.utc)
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing

            new_doc = IngestedSourceDocument(
                file_sha=file_sha,
                file_name=file_name,
                provider=provider,
                account_country=account_country,
                status=status_str,
                transaction_count=transaction_count,
                error_message=error_message,
                processed_at=datetime.now(timezone.utc),
            )
            session.add(new_doc)
            session.commit()
            session.refresh(new_doc)
            return new_doc

    # --- FinancialRecord helper methods using SQLModel session ---

    def _update_financial_record(self, record: FinancialRecord) -> FinancialRecord:
        """Update a transaction record using SQLModel."""
        with Session(self.engine) as session:
            merged = session.merge(record)
            session.commit()
            session.refresh(merged)
            return merged

    def update_strict_financial_record(self, strict_record: BaseStrictRecord) -> BaseStrictRecord:
        """Update a transaction record using a validated domain model."""
        raw_record = strict_record.to_raw()
        updated_raw = self._update_financial_record(raw_record)
        return BaseStrictRecord.from_raw(updated_raw)

    @deprecated("Use get_strict_financial_record instead to receive a validated domain model")
    def get_financial_record_by_id(self, record_id: int) -> FinancialRecord | None:
        """Fetch a specific transaction record by its primary key ID."""
        with Session(self.engine) as session:
            return session.get(FinancialRecord, record_id)

    def get_strict_financial_record(self, record_id: int) -> BaseStrictRecord | None:
        """Fetch a specific transaction record by its primary key ID and return a validated domain model."""
        raw = self.get_financial_record_by_id(record_id)
        if raw is None:
            return None
        return BaseStrictRecord.from_raw(raw)

    @deprecated("use BaseStrictRecord.from_raw to validate and convert FinancialRecord instead")
    def get_financial_records(
        self,
        account_country: str | None = None,
        tax_year: int | None = None,
        verification_status: str | None = None,
        source_file_name: str | None = None,
        source_file_sha: str | None = None,
    ) -> list[FinancialRecord]:
        """Query transaction records with optional filters.

        Note:
            Deprecated for direct business logic evaluation. Prefer using validated
            domain model getters (e.g. ``get_validated_buy_records_by_isin``) or
            instantiating ``BaseStrictRecord.from_raw`` to ensure strict domain invariant checking.
        """
        with Session(self.engine) as session:
            statement = select(FinancialRecord)
            if account_country:
                statement = statement.where(FinancialRecord.account_country == account_country)
            if tax_year:
                statement = statement.where(FinancialRecord.tax_year == tax_year)
            if verification_status:
                statement = statement.where(FinancialRecord.verification_status == verification_status)
            if source_file_name:
                statement = statement.where(FinancialRecord.source_file_name == source_file_name)
            if source_file_sha:
                statement = statement.where(FinancialRecord.source_file_sha == source_file_sha)
            statement = statement.order_by(text("event_timestamp"))
            return list(session.exec(statement).all())

    # --- StagedFinancialRecord helper methods using SQLModel session ---

    def insert_staged_record(self, record: StagedFinancialRecord) -> StagedFinancialRecord:
        """Insert a staged transaction record using SQLModel."""
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def update_staged_record(self, record: StagedFinancialRecord) -> StagedFinancialRecord:
        """Update a staged transaction record using SQLModel."""
        with Session(self.engine) as session:
            merged = session.merge(record)
            session.commit()
            session.refresh(merged)
            return merged

    def get_staged_record_by_id(self, record_id: int) -> StagedFinancialRecord | None:
        """Fetch a specific staged transaction record by its primary key ID."""
        with Session(self.engine) as session:
            return session.get(StagedFinancialRecord, record_id)

    def get_staged_records(
        self,
        account_country: str | None = None,
        verification_status: str | None = None,
        source_file_name: str | None = None,
        source_file_sha: str | None = None,
    ) -> list[StagedFinancialRecord]:
        """Query staged transaction records with optional filters."""
        with Session(self.engine) as session:
            statement = select(StagedFinancialRecord)
            if account_country:
                statement = statement.where(StagedFinancialRecord.account_country == account_country)
            if verification_status:
                statement = statement.where(StagedFinancialRecord.verification_status == verification_status)
            if source_file_name:
                statement = statement.where(StagedFinancialRecord.source_file_name == source_file_name)
            if source_file_sha:
                statement = statement.where(StagedFinancialRecord.source_file_sha == source_file_sha)
            statement = statement.order_by(text("event_timestamp"))
            return list(session.exec(statement).all())

    def delete_staged_record_by_id(self, record_id: int) -> bool:
        """Delete a single staged financial record by ID. Returns True if deleted, False if not found."""
        with Session(self.engine) as session:
            record = session.get(StagedFinancialRecord, record_id)
            if record:
                session.delete(record)
                session.commit()
                return True
            return False

    def _find_duplicate_financial_records(self, staged: StagedFinancialRecord) -> list[FinancialRecord]:
        """Find potential duplicate records in financial_records matching asset, quantity, action, and date."""
        if not staged.event_timestamp or not staged.action or staged.quantity is None:
            return []

        start_of_day = datetime.combine(staged.event_timestamp.date(), time.min)
        end_of_day = datetime.combine(staged.event_timestamp.date(), time.max)
        act_str = staged.action.strip().lower()

        with Session(self.engine) as session:
            statement = select(FinancialRecord).where(
                col(FinancialRecord.event_timestamp) >= start_of_day,
                col(FinancialRecord.event_timestamp) <= end_of_day,
                func.lower(col(FinancialRecord.action)) == act_str,
                col(FinancialRecord.quantity) == staged.quantity,
            )

            if staged.isin:
                statement = statement.where(func.lower(col(FinancialRecord.isin)) == staged.isin.strip().lower())
            elif staged.symbol:
                statement = statement.where(func.lower(col(FinancialRecord.symbol)) == staged.symbol.strip().lower())
            elif staged.asset_type:
                statement = statement.where(
                    func.lower(col(FinancialRecord.asset_type)) == staged.asset_type.strip().lower()
                )

            return list(session.exec(statement).all())

    def approve_staged_record(
        self,
        staged_id: int,
        record: BaseStrictRecord | None = None,
        force_duplicate: bool = False,
    ) -> tuple[bool, str, FinancialRecord | None, list[FinancialRecord]]:
        """Approve a staged record by persisting validated domain model and linking backtracking ID.

        Args:
            staged_id: Primary key ID of the staged record to approve.
            record: Optional pre-validated BaseStrictRecord domain model.
            force_duplicate: If True, bypass potential duplicate checks.

        Returns:
            Tuple of (success, message, created_financial_record, duplicates).
        """
        with Session(self.engine) as session:
            staged = session.get(StagedFinancialRecord, staged_id)
            if not staged:
                return False, "Staged record not found", None, []

            missing = staged.get_missing_fields()
            if missing:
                return False, f"Missing required fields: {', '.join(missing)}", None, []

            duplicates = self._find_duplicate_financial_records(staged)
            if duplicates and not force_duplicate:
                return False, "potential_duplicate", None, duplicates

            if record is not None:
                financial_rec = record.to_raw()
            else:
                financial_rec = FinancialRecord(
                    provider=staged.provider,
                    source_file_name=staged.source_file_name,
                    source_file_sha=staged.source_file_sha,
                    event_timestamp=staged.event_timestamp,
                    ingestion_timestamp=staged.ingestion_timestamp,
                    asset_type=staged.asset_type,
                    symbol=staged.symbol,
                    isin=staged.isin,
                    asset_name=staged.asset_name,
                    action=staged.action,
                    quantity=staged.quantity,
                    unit_price=staged.unit_price,
                    currency=staged.currency,
                    fees=staged.fees,
                    total_amount=staged.total_amount,
                    fx_rate=staged.fx_rate,
                    local_total_amount=staged.local_total_amount,
                    tax_year=staged.tax_year,
                    account_country=staged.account_country,
                    additional_metadata=staged.additional_metadata,
                    verification_status="approved",
                    consensus_log=staged.consensus_log,
                    openfigi_detected=staged.openfigi_detected,
                )

            session.add(financial_rec)
            session.flush()

            staged.verification_status = "approved"
            staged.approved_financial_record_id = financial_rec.id
            session.merge(staged)
            session.commit()
            session.refresh(financial_rec)
            return True, "approved", financial_rec, []

    def reject_staged_record(self, staged_id: int) -> bool:
        """Reject a staged transaction record.

        Args:
            staged_id: Primary key ID of the staged record to reject.

        Returns:
            True if record was found and marked as rejected, False if record not found.
        """
        with Session(self.engine) as session:
            staged = session.get(StagedFinancialRecord, staged_id)
            if not staged:
                return False
            staged.verification_status = "rejected"
            session.merge(staged)
            session.commit()
            return True

    def filter_financial_records(  # noqa: PLR0917
        self,
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
        limit: int | None = None,
    ) -> list[BaseStrictRecord]:
        """Filter approved financial records and return validated BaseStrictRecord domain models.

        Args:
            asset_type: Optional asset type filter (e.g. 'stock', 'etf', 'cash').
            action: Optional transaction action filter (e.g. 'buy', 'sell', 'dividend').
            tax_year: Optional tax year filter (e.g. 2025).
            isin: Optional ISIN identifier filter.
            quantity_over: Optional lower bound Decimal for quantity (quantity >= quantity_over).
            quantity_less: Optional upper bound Decimal for quantity (quantity <= quantity_less).
            purchase_date_start: Optional start datetime timestamp (event_timestamp >= start).
            purchase_date_end: Optional end datetime timestamp (event_timestamp <= end).
            logic: Logical operator for combining filter conditions ('AND' or 'OR').
            account_country: Optional account country filter ('italy', 'ireland').
            limit: Optional maximum records to return (None for unlimited).

        Returns:
            List of validated BaseStrictRecord domain instances ordered by event_timestamp.
        """
        conditions: list[ColumnElement[bool]] = []
        if asset_type:
            conditions.append(func.lower(col(FinancialRecord.asset_type)) == asset_type.strip().lower())
        if action:
            conditions.append(func.lower(col(FinancialRecord.action)) == action.strip().lower())
        if tax_year is not None:
            conditions.append(col(FinancialRecord.tax_year) == tax_year)
        if isin:
            conditions.append(func.lower(col(FinancialRecord.isin)) == isin.strip().lower())
        if account_country:
            conditions.append(col(FinancialRecord.account_country) == account_country.strip())
        if quantity_over is not None:
            conditions.append(col(FinancialRecord.quantity) >= quantity_over)
        if quantity_less is not None:
            conditions.append(col(FinancialRecord.quantity) <= quantity_less)
        if purchase_date_start is not None:
            conditions.append(col(FinancialRecord.event_timestamp) >= purchase_date_start)
        if purchase_date_end is not None:
            conditions.append(col(FinancialRecord.event_timestamp) <= purchase_date_end)

        with Session(self.engine) as session:
            statement = select(FinancialRecord)
            if conditions:
                if logic.upper() == "OR":
                    statement = statement.where(or_(*conditions))
                else:
                    statement = statement.where(and_(*conditions))

            statement = statement.order_by(text("event_timestamp"))
            if limit is not None:
                statement = statement.limit(limit)

            raw_records = list(session.exec(statement).all())
            return [BaseStrictRecord.from_raw(r) for r in raw_records]

    def delete_financial_records_by_provider(self, provider: str, account_country: str) -> None:
        """Remove all transaction records from a specific provider and account country."""
        with Session(self.engine) as session:
            statement = select(FinancialRecord).where(
                FinancialRecord.provider == provider, FinancialRecord.account_country == account_country
            )
            records = session.exec(statement).all()
            for record in records:
                session.delete(record)
            session.commit()
            print(f"Deleted {len(records)} financial records for provider '{provider}' ({account_country}).")

    def delete_financial_record_by_id(self, record_id: int) -> bool:
        """Delete a single financial record by ID. Returns True if deleted, False if not found."""
        with Session(self.engine) as session:
            record = session.get(FinancialRecord, record_id)
            if record:
                session.delete(record)
                session.commit()
                return True
            return False

    # --- CGT Engine Helper Methods ---

    def get_taxpayer_profile(self, tax_year: int) -> TaxpayerProfile | None:
        """Fetch taxpayer profile by tax year."""
        with Session(self.engine) as session:
            statement = select(TaxpayerProfile).where(TaxpayerProfile.tax_year == tax_year)
            return session.exec(statement).first()

    def get_all_taxpayer_profiles(self) -> list[TaxpayerProfile]:
        """Fetch all taxpayer profile records across all tax years."""
        with Session(self.engine) as session:
            statement = select(TaxpayerProfile).order_by(text("tax_year DESC"))
            return list(session.exec(statement).all())

    def upsert_taxpayer_profile(self, profile: TaxpayerProfile) -> TaxpayerProfile:
        """Insert or update a taxpayer profile enforcing tax_year uniqueness.

        Args:
            profile: TaxpayerProfile instance to persist.

        Returns:
            Persisted TaxpayerProfile instance.

        Raises:
            ValueError: If a profile for the same tax_year already exists with a different ID.
        """
        existing = self.get_taxpayer_profile(profile.tax_year)
        if existing and profile.id is not None and existing.id != profile.id:
            raise ValueError(f"TaxpayerProfile already exists for tax_year {profile.tax_year} (id={existing.id}).")

        with Session(self.engine) as session:
            merged = session.merge(profile)
            session.commit()
            session.refresh(merged)
            return merged

    def get_asset_tax_classification(self, isin: str) -> AssetTaxClassification | None:
        """Fetch asset tax classification by ISIN."""
        with Session(self.engine) as session:
            return session.get(AssetTaxClassification, isin)

    def upsert_asset_tax_classification(self, classification: AssetTaxClassification) -> AssetTaxClassification:
        """Insert or update asset tax classification using SQLModel."""
        with Session(self.engine) as session:
            merged = session.merge(classification)
            session.commit()
            session.refresh(merged)
            return merged

    def get_remittance_events(self, financial_record_id: int) -> list[RemittanceEvent]:
        """Fetch all remittance events for a financial record ID."""
        with Session(self.engine) as session:
            statement = (
                select(RemittanceEvent)
                .where(RemittanceEvent.financial_record_id == financial_record_id)
                .order_by(col(RemittanceEvent.remittance_date))
            )
            return list(session.exec(statement).all())

    def add_remittance_event(self, event: RemittanceEvent) -> RemittanceEvent:
        """Add a remittance event record."""
        with Session(self.engine) as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def _get_buy_records_by_isin(self, isin: str) -> list[FinancialRecord]:
        """Fetch all buy records for an ISIN across all account_countrys ordered by timestamp."""
        with Session(self.engine) as session:
            statement = (
                select(FinancialRecord)
                .where(func.lower(FinancialRecord.isin) == isin.lower(), func.lower(FinancialRecord.action) == "buy")
                .order_by(col(FinancialRecord.event_timestamp))
            )
            return list(session.exec(statement).all())

    def _get_sell_records_by_isin(self, isin: str) -> list[FinancialRecord]:
        """Fetch all sell records for an ISIN across all account_countrys ordered by timestamp."""
        with Session(self.engine) as session:
            statement = (
                select(FinancialRecord)
                .where(func.lower(FinancialRecord.isin) == isin.lower(), func.lower(FinancialRecord.action) == "sell")
                .order_by(col(FinancialRecord.event_timestamp))
            )
            return list(session.exec(statement).all())

    def _get_purchases_in_window(self, isin: str, after: datetime, before: datetime) -> list[FinancialRecord]:
        """Fetch all buy records for an ISIN in a date window (after, before] across all account_countrys."""
        with Session(self.engine) as session:
            statement = (
                select(FinancialRecord)
                .where(
                    func.lower(FinancialRecord.isin) == isin.lower(),
                    func.lower(FinancialRecord.action) == "buy",
                    col(FinancialRecord.event_timestamp) > after,
                    col(FinancialRecord.event_timestamp) <= before,
                )
                .order_by(col(FinancialRecord.event_timestamp))
            )
            return list(session.exec(statement).all())

    def get_validated_buy_records_by_isin(self, isin: str) -> list[TradeRecord]:
        """Fetch and validate all buy TradeRecord instances for an ISIN ordered by timestamp.

        Note:
            Rows in ``financial_records`` table are approved ledger entries promoted from staging.
            Instantiating ``BaseStrictRecord.from_raw`` validates TradeRecord domain invariants
            (e.g., positive quantities and currency formatting) via Pydantic model validation.

        Raises:
            ValueError: If a record fails validation or cannot be converted to a valid TradeRecord.
        """
        raw_records = self._get_buy_records_by_isin(isin)
        trade_records: list[TradeRecord] = []
        for record in raw_records:
            strict = BaseStrictRecord.from_raw(record)
            if not isinstance(strict, TradeRecord):
                raise ValueError(f"Financial record {record.id} cannot be converted to TradeRecord.")
            trade_records.append(strict)
        trade_records.sort(key=lambda r: r.event_timestamp)
        return trade_records

    def get_validated_sell_records_by_isin(self, isin: str) -> list[TradeRecord]:
        """Fetch and validate all sell TradeRecord instances for an ISIN ordered by timestamp.

        Note:
            Rows in ``financial_records`` table are approved ledger entries promoted from staging.
            Instantiating ``BaseStrictRecord.from_raw`` validates TradeRecord domain invariants
            (e.g., positive quantities and currency formatting) via Pydantic model validation.

        Raises:
            ValueError: If a record fails validation or cannot be converted to a valid TradeRecord.
        """
        raw_records = self._get_sell_records_by_isin(isin)
        trade_records: list[TradeRecord] = []
        for record in raw_records:
            strict = BaseStrictRecord.from_raw(record)
            if not isinstance(strict, TradeRecord):
                raise ValueError(f"Financial record {record.id} cannot be converted to TradeRecord.")
            trade_records.append(strict)
        trade_records.sort(key=lambda r: r.event_timestamp)
        return trade_records

    def get_purchases_in_window(self, isin: str, after: datetime, before: datetime) -> list[TradeRecord]:
        """Fetch and validate buy TradeRecord instances for an ISIN in a date window (after, before].

        Note:
            Rows in ``financial_records`` table are approved ledger entries promoted from staging.
            Instantiating ``BaseStrictRecord.from_raw`` validates TradeRecord domain invariants
            (e.g., positive quantities and currency formatting) via Pydantic model validation.

        Raises:
            ValueError: If a record fails validation or cannot be converted to a valid TradeRecord.
        """
        raw_records = self._get_purchases_in_window(isin, after, before)
        trade_records: list[TradeRecord] = []
        for record in raw_records:
            strict = BaseStrictRecord.from_raw(record)
            if not isinstance(strict, TradeRecord):
                raise ValueError(f"Financial record {record.id} cannot be converted to TradeRecord.")
            trade_records.append(strict)
        trade_records.sort(key=lambda r: r.event_timestamp)
        return trade_records

    def get_isins_by_regime(self, tax_regime: str) -> list[str]:
        """Fetch all distinct ISINs classified under a specific tax regime."""
        with Session(self.engine) as session:
            statement = select(AssetTaxClassification.isin).where(AssetTaxClassification.tax_regime == tax_regime)
            return list(session.exec(statement).all())

    # --- AssetMerger / Corporate Action helper methods ---

    def insert_asset_merger(self, merger: AssetMerger) -> AssetMerger:
        """Insert or update an ETF corporate merger record.

        Raises:
            ValueError: If a merger for the same old ISIN already exists with a different ID.
        """
        existing = self.get_merger_by_old_isin(merger.old_isin)
        if existing and existing.id != merger.id:
            raise ValueError(f"AssetMerger already exists for old_isin '{merger.old_isin}' (id={existing.id}).")

        with Session(self.engine) as session:
            session.add(merger)
            session.commit()
            # session.refresh(merger) reloads database-generated primary key 'id' and 'created_at' into memory post-commit.
            session.refresh(merger)
            return merger

    def get_asset_mergers(self) -> list[AssetMerger]:
        """Fetch all registered ETF corporate mergers ordered by effective date."""
        with Session(self.engine) as session:
            statement = select(AssetMerger).order_by(col(AssetMerger.effective_date))
            return list(session.exec(statement).all())

    def get_merger_by_old_isin(self, old_isin: str) -> AssetMerger | None:
        """Fetch corporate merger mapping for a specific old ISIN.

        Args:
            old_isin: Target old ISIN string identifier.

        Returns:
            Matching AssetMerger instance if registered, otherwise None.
        """
        with Session(self.engine) as session:
            statement = (
                select(AssetMerger)
                .where(func.lower(col(AssetMerger.old_isin)) == old_isin.strip().lower())
                .order_by(col(AssetMerger.effective_date).desc())
            )
            return session.exec(statement).first()

    def delete_asset_merger(self, merger_id: int) -> bool:
        """Delete an asset merger record by ID."""
        with Session(self.engine) as session:
            record = session.get(AssetMerger, merger_id)
            if record:
                session.delete(record)
                session.commit()
                return True
            return False

    def insert_tax_income_record(self, record: StrictTaxIncomeRecord) -> int:
        """Insert a tax income record into the relational database.

        Args:
            record: Validated StrictTaxIncomeRecord domain model instance.

        Returns:
            Assigned primary key ID.
        """
        raw = record.to_raw()
        with Session(self.engine) as session:
            session.add(raw)
            session.commit()
            session.refresh(raw)
            assert raw.id is not None
            return raw.id

    def get_tax_income_records(
        self,
        tax_year: int | None = None,
        jurisdiction: str | None = None,
    ) -> list[StrictTaxIncomeRecord]:
        """Fetch tax income records and return validated StrictTaxIncomeRecord domain models.

        Args:
            tax_year: Optional tax year filter.
            jurisdiction: Optional jurisdiction filter ('ireland', 'italy', etc.).

        Returns:
            List of validated StrictTaxIncomeRecord domain instances.
        """
        with Session(self.engine) as session:
            statement = select(TaxIncomeRecord)
            if tax_year is not None:
                statement = statement.where(TaxIncomeRecord.tax_year == tax_year)
            if jurisdiction:
                statement = statement.where(
                    func.lower(col(TaxIncomeRecord.jurisdiction)) == jurisdiction.strip().lower()
                )
            statement = statement.order_by(col(TaxIncomeRecord.tax_year).desc())
            raw_records = list(session.exec(statement).all())
            return [StrictTaxIncomeRecord.from_raw(r) for r in raw_records]

    def delete_tax_income_record(self, record_id: int) -> bool:
        """Delete a tax income record by primary key ID."""
        with Session(self.engine) as session:
            record = session.get(TaxIncomeRecord, record_id)
            if record:
                session.delete(record)
                session.commit()
                return True
            return False

    @property
    def conn(self) -> PoolProxiedConnection:
        """Provide a raw connection helper for backward compatibility with legacy scripts."""
        return self.engine.raw_connection()

    def close(self):
        """Close connection engine."""
        self.engine.dispose()
