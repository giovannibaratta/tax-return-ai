# Tax-Return-AI: Cross-Border Tax Ingestion & Computation (Italy & Ireland)

A developer-first, local-only multi-agent system designed to digest cross-border regulatory guides and financial statements for personal tax preparation (Tax Year 2025).

For detailed pipeline flows, component diagrams, and data models, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ⚡ Quick Commands Cheat Sheet

| Task | Command |
|---|---|
| **Run All Unit Tests** | `pytest` |
| **Ingest Regulatory Documents** | `python backend/ingestion/ingest.py` |
| **Ingest Broker & Income Records** | `python backend/ingestion/ingest_transactions.py ingest` |
| **Launch Desktop UI Application** | `python -m src.ui.main` |
| **Start Local MCP Server** | `python backend/mcp_server.py` |

---

## 🛠️ Installation & Environment Setup

The application requires **Python 3.10+** (Python 3.11+ recommended).

### 1. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pre-download NLP Tokenizers (Stanza)
Sentence boundary detection relies on Stanza tokenizer models for Italian (`it`) and English (`en`):
```bash
python -c "import stanza; stanza.download('it', verbose=False); stanza.download('en', verbose=False)"
```

### 4. Verify Setup
```bash
pytest
```

---

## 🚀 Running Ingestion

### 1. Regulatory Documents Ingestion
Scans `data/raw_sources/` and indexes regulatory tax PDFs using Late Chunking embeddings:

```bash
# Ingest newly added documents
python backend/ingestion/ingest.py

# Force re-indexing all documents
python backend/ingestion/ingest.py --force

# Ingest a single document
python backend/ingestion/ingest.py --file data/raw_sources/italy/testo_unico_imposte_redditi.pdf
```

### 2. Financial & Income Records Ingestion
Scans `data/raw_sources/records/` and runs multi-voter LLM consensus:

```bash
python backend/ingestion/ingest_transactions.py ingest
```

#### CLI Options:

| Flag | Description | Values |
|---|---|---|
| `--mode` | Execution mode for LLM runners | `local` (default), `api`, `interactive`, `mock` |
| `--parser` | Extraction backend parser | `chandra` (default), `pdfplumber`, `chandra_api` |
| `--force-ocr` | Bypass local OCR cache and re-run OCR | Boolean flag |
| `--force-pii` | Bypass local PII cache and re-sanitize | Boolean flag |
| `--force` | Force re-ingestion and record replacement | Boolean flag |
| `--test` / `-t` | Required opt-in flag for non-production modes | Boolean flag |
| `--db` | Custom SQLite database path override | File path |

### 3. Listing & Managing Ingested Records
```bash
# List all persisted financial records
python backend/ingestion/ingest_transactions.py list

# Delete a specific record by ID
python backend/ingestion/ingest_transactions.py delete --id <id>

# Clear all records
python backend/ingestion/ingest_transactions.py delete --all
```

---

## 💻 Local Inference & Hardware Acceleration

To configure local LLMs (Ollama, llama-server, vLLM) or hardware acceleration on Apple Silicon (Metal/MPS), refer to the [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md) guide.

---

## 🔌 Model Context Protocol (MCP) Server Integration

The project exposes an MCP server using `FastMCP` over `stdio` transport, enabling external AI assistants (Cursor, Claude Desktop) to query the tax knowledge base and financial records.

### Exposed Operations:
- **`list_documents`**: Lists all indexed regulatory manuals and documents.
- **`query_tax_knowledge(query_text, limit, jurisdiction)`**: High-precision vector semantic search over regulatory chunks.
- **`keyword_search_knowledge(keyword, limit, jurisdiction)`**: Fast text matching over indexed chunks.
- **`get_chunk_content(chunk_id)`**: Retrieves markdown text and metadata for a specific chunk.
- **`get_surrounding_context(document_name, chunk_index, before, after)`**: Fetches neighboring chunks for expanded context.
- **`get_document_statistics(document_name)`**: Returns page and chunk counts for a document.

### Claude Desktop / Cursor Configuration Example:
```json
{
  "mcpServers": {
    "tax-compliance-context": {
      "command": "/path/to/tax-return-ai/.venv/bin/python3",
      "args": [
        "/path/to/tax-return-ai/backend/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/tax-return-ai"
      }
    }
  }
}
```

---

## 📖 Development & Testing Guidelines

For database migration instructions (Liquibase) and test conventions, see [DEVELOPMENT.md](DEVELOPMENT.md) and [AGENTS.md](AGENTS.md).
