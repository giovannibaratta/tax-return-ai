# Tax-Return-AI: Cross-Border Tax Ingestion & Computation (Italy & Ireland)

A developer-first, local-only multi-agent system designed to digest cross-border regulatory guides and statements for personal tax preparation (Tax Year 2025). 

(TODO): This should be moved to an Architecture.md file. leaving the README only for installation and usage instructions.

This repository implements **Seasonal Batch Ingestion** using page-by-page PDF table layout extraction, **Late Chunking** token-pooling via a local **BGE-M3** model, and **`sqlite-vec`** vector storage in a single SQLite database.

---

## 🏗️ Architectural Overview
(TODO): This should be a mermaid diagram
```
                      (TODO): The data structure have been changed a bit and has more sources
                      +---------------------------------------+
                      |       Seasonal Drop Folder            |
                      |   data/raw_sources/{italy,ireland}   |
                      +-------------------+-------------------+
                                          |
                                          v. (TODO): pdfplumber is the fallback, we can keep it more highlevel
                      +-------------------+-------------------+
                      |   pdfplumber Layout & Table Extractor |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |   Local BGE-M3 Late Chunker (MPS/GPU) |
                      +-------------------+-------------------+
                                          |
                                          v (TODO): no need to mention the low level tables
                      +-------------------+-------------------+
                      |     database/tax_data.db (SQLite)     |
                      |  - tax_document_metadata              |
                      |  - vss_tax_chunks (sqlite-vec)        |
                      +-------------------+-------------------+
                                          |
                                          v (TODO): We can expand by adding tools (maybe 3 differente diagrams where this become just a regulation pipeline, then we have the financial records pipeline, then another one that show the tools aggregated together)
                      +-------------------+-------------------+
                      |  Semantic Query & Traceability Layer   |
                      +---------------------------------------+
```

---

## 🛠️ Installation & Environment Setup

The application requires **Python 3.10+** (Python 3.11+ recommended) and supports both local hardware acceleration (Apple Silicon MPS / CUDA) and remote/API-based LLMs.

Follow these steps to set up a clean development workspace:

### 1. Set Up the Virtual Environment
Create and activate an isolated Python virtual environment inside the project root:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install all core dependencies including PyTorch, Sentence Transformers, `stanza`, Pydantic, SQLModel, and `sqlite-vec`:
```bash
pip install -r requirements.txt
```

### 3. Pre-download Prerequisite Language Models (Stanza NLP)
Sentence boundary detection relies on pre-downloaded Stanza tokenizer models for Italian (`it`) and English (`en`). Pre-download these models as a prerequisite step so test suite operate fully offline:
```bash
python -c "import stanza; stanza.download('it', verbose=False); stanza.download('en', verbose=False)"
```

### 4. Verify Workspace Setup (Run Tests)
Run the unit test suite using `pytest` to verify the workspace is fully functional:
```bash
pytest
```

---

(TODO): we should have a top section with a recap of all the commands/scripts without all the variation, just the exptend commnad to run in most of the cases.

## 🚀 Running the Ingestion Layer

The batch orchestrator automatically scans the drop folders, processes new files page-by-page, and indexes them incrementally.

### 1. Run Ingestion (Batch Mode)
To scan `data/raw_sources/` and ingest any newly added or updated tax PDFs:
```bash
python backend/ingestion/ingest.py
```

### 2. Force Re-index All Documents
By default, the script skips documents that are already loaded in SQLite. To wipe existing indexes and force a full recalculation of all files, add the `--force` flag:
```bash
python backend/ingestion/ingest.py --force
```

### 3. Ingest a Single PDF File
To process and index only one specific regulatory manual:
```bash
python backend/ingestion/ingest.py --file data/raw_sources/italy/testo_unico_imposte_redditi/testo_unico_imposte_redditi_articolo_67.pdf
```

### 4. Running Transaction Ingestion (Consensus Pipeline)
To scan `data/raw_sources/records/` and ingest portfolio/transaction reports (Directa, Interactive Brokers, F24 tax returns) using three Voter Agents:
```bash
python backend/ingestion/ingest_transactions.py ingest --force --parser chandra
```
(TODO): table might be better ?
- `--mode`: Select execution mode (`api` / `local` [default, uses `.env` LLM config], `interactive`, `mock`).
- `--test` / `-t`: Explicit opt-in flag required when running non-production test/mock modes (`--mode mock --test`).
- `--parser`: Select extraction backend parser (`chandra` [default], `pdfplumber`, `chandra_api`).
- `--force-ocr`: Force re-running OCR parsing from scratch, bypassing local disk OCR cache.
- `--force`: Force re-ingestion and database record replacement.

(TODO): Is it still marked as approved ?
This runs three Voter Agents separately. If they all agree on all fields, the transaction is marked as `APPROVED`. If they disagree, it is marked as `ESCALATED_TO_USER` (resolvable in the UI).

### 5. Listing Ingested Transactions
To view a list of all transactions and factual records stored in the database:
```bash
python backend/ingestion/ingest_transactions.py list
```

(TODO): This should be more generic (not production vs development but something else). we can mention in DEVELOPMENT.md as a mechanism for testing.
(TODO): This should be supported by all the scripts if it is not already the case.
### 6. Separating "Production" vs "Development" Databases
You can direct database operations to a different database file (e.g. for developer isolation or testing) using either:
- **Environment Variable**:
  ```bash
  export TAX_DB_PATH="database/tax_data_dev.db"
  python backend/ingestion/ingest_transactions.py ingest
  ```
- **CLI Flag**:
  ```bash
  python backend/ingestion/ingest_transactions.py ingest --db database/tax_data_dev.db
  ```

### 7. Deleting Records
- **Delete a single transaction record** (requires user confirmation):
  ```bash
  python backend/ingestion/ingest_transactions.py delete --id <id>
  ```
- **Clear all financial transaction records** (requires user confirmation):
  ```bash
  python backend/ingestion/ingest_transactions.py delete --all
  ```

(TODO): Doesn't this applies to all the scripts in general ? The setup, while the multi-model consensus is specific to this script.
### 9. Setting Up Local / Real LLMs & OCR
For detailed instructions on configuring PyTorch GPU acceleration (MPS), HuggingFace cache directories, Ollama commands, and `sqlite-vec` native extensions locally, refer to the [LOCAL_MODEL_SETUP.md](file:///Users/gio/repos/tax-return-ai/LOCAL_MODEL_SETUP.md) guide.

(TODO): This still reference pdfplumber has default mechanism.
The ingestion pipeline extracts text and layout tables natively from PDFs using `pdfplumber`. To run the consensus engine with real LLMs, you can create a local `.env` file in the project root containing your configurations:

#### Option A: Single Model / Default Model Fallback
If you want all three voter agents to use the same model, define:
```env
DEBATE_LLM_BASE_URL="http://localhost:11434/v1" # e.g. local Ollama URL
DEBATE_LLM_API_KEY="ollama"                    # Required non-empty API Key
DEBATE_LLM_MODEL="gemma"                       # Model name
```

(TODO): Do we still have agents with a different focus (chronological, aritmehtic, ...)
#### Option B: Multi-Model Voter Consensus Configuration
To run judge-style consensus with different models (e.g., Voter 1 runs `gemma` on Ollama, Voter 2 runs `llama3` on Ollama, and Voter 3 runs `gpt-4o-mini` on OpenRouter), define individual voter configurations in your `.env` file:
```env
# Voter 1 (chronological dates focus) - Local Ollama Gemma
VOTER_1_BASE_URL="http://localhost:11434/v1"
VOTER_1_API_KEY="ollama"
VOTER_1_MODEL="gemma"

# Voter 2 (arithmetic totals focus) - Local Ollama Llama3
VOTER_2_BASE_URL="http://localhost:11434/v1"
VOTER_2_API_KEY="ollama"
VOTER_2_MODEL="llama3"

# Voter 3 (asset classification focus) - Remote OpenRouter GPT-4o-mini
VOTER_3_BASE_URL="https://openrouter.ai/api/v1"
VOTER_3_API_KEY="your-openrouter-api-key"
VOTER_3_MODEL="openai/gpt-4o-mini"
```

Once configured, run the ingestion in `api` mode:
```bash
python backend/ingestion/ingest_transactions.py --mode api
```

#### Option C: Chandra OCR Parsing Configuration
Both `ingest.py` and `ingest_transactions.py` support layout-aware OCR parsing via **Chandra OCR-2** (`--parser chandra` or `--parser chandra_api`).

Configurations in `.env`:
- **Local PyTorch / HuggingFace** (default local execution):
  ```env
  CHANDRA_INFERENCE_METHOD="hf"   # 'hf' for local PyTorch or 'vllm' for remote vLLM server
  TORCH_DEVICE="mps"              # 'mps', 'cuda', or 'cpu' (auto-detected if omitted)
  TORCH_ATTN="sdpa"               # PyTorch Scaled Dot-Product Attention optimization
  ```
- **Remote vLLM Server**:
  ```env
  CHANDRA_INFERENCE_METHOD="vllm"
  VLLM_API_BASE="http://192.168.1.100:8000/v1"
  VLLM_API_KEY="your-vllm-api-key"
  VLLM_MODEL_NAME="datalab-to/chandra-ocr-2"
  ```
- **Datalab Cloud API** (`--parser chandra_api`):
  ```env
  DATALAB_API_KEY="your-datalab-api-key"
  DATALAB_MODE="balanced"          # 'balanced', 'fast', or 'accurate'
  ```

---

## 🔍 Querying the Vector Database (Semantic Search)

The system stores both text content and 1024-dimensional vectors in a single, local SQLite file located at `database/tax_data.db`.

(TODO): We cannot reference my local gemini dir, either we drop or define a proper script.
### 1. Verification Script
We have included a pre-configured search test script in your brain scratchpad. To execute semantic test queries against the active database, run:
```bash
python /Users/gio/.gemini/antigravity-ide/brain/c596ec3b-ce8c-479f-964f-4797fb23f230/scratch/verify_search.py
```
(TODO): I don't think this should be here at all.
### 2. Custom SQL Query Pattern (python)
To run vector similarity searches inside your own modules, use standard Python `sqlite3` and the `sqlite-vec` `MATCH` syntax:

```python
import sqlite3
import sqlite_vec

# 1. Connect and Load Vector Extension
conn = sqlite3.connect("database/tax_data.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

# 2. Serialize Query Vector
# query_embedding must be a List[float] of length 1024 (from BGE-M3)
serialized_query = sqlite_vec.serialize_float32(query_embedding)

# 3. Execute KNN Search (e.g. limit=5)
cursor = conn.cursor()
sql = """
    WITH matches AS (
        SELECT chunk_id, distance
        FROM vss_tax_chunks
        WHERE embedding MATCH ? AND k = 5
    )
    SELECT m.distance, t.document_name, t.page_number, t.text_content
    FROM matches m
    JOIN tax_document_metadata t ON m.chunk_id = t.id
    WHERE t.jurisdiction = 'italy'  -- Optional filtering
"""
cursor.execute(sql, (serialized_query,))
for row in cursor.fetchall():
    print(f"Dist: {row[0]:.4f} | Page: {row[2]} | {row[3][:100]}...")
```

---

(TODO): Can we dropped.
## 🗄️ Database Schema Design

The SQLite database (`database/tax_data.db`) stores document metadata, vector chunks, and financial records. For the exact schema, types, and up-to-date field definitions, please refer directly to the source code definitions in [backend/models.py](file:///Users/gio/repos/tax-return-ai/backend/models.py).

---
(TODO): We already have a section for multi-consesus voter model. Either we aggreagte or we define something under docs/
## 🤖 Multi-Agent Consensus Voter Models

The transaction ingestion pipeline employs a three-agent voter consensus logic to guarantee absolute mathematical accuracy of ingested data:
*   **Voter Agent 1**: Focuses on chronological ordering and precise date/time parsing.
*   **Voter Agent 2**: Focuses on strict arithmetic constraints (`quantity * unit_price + fees = total_amount`).
*   **Voter Agent 3**: Focuses on correct asset type categorization (e.g., stock vs. ETF) and transaction actions.

### Simulated Voter Configurations in Tests
During testing and mock mode execution, the pipeline runs the **`MockRunner`** simulating these three agents:
*   **Directa Ingestion Test**: Simulates unanimous voter consensus. All 3 voter agents agree on all fields. The transaction is saved as `APPROVED`.
*   **IBKR Ingestion Test**: Simulates voter disagreement (Voter 3 extracts a mismatching quantity). The pipeline escalates the record, saving it as `ESCALATED_TO_USER` to prevent silent data errors and prompt interactive CLI verification.

---

## 🔌 Model Context Protocol (MCP) Server Integration

The project exposes a highly optimized **Model Context Protocol (MCP)** server built with `FastMCP` running over standard I/O (`stdio`) transport. This allows other AI agents or systems (like Claude Desktop or Cursor) to interact with your vector-indexed regulatory database dynamically.

(TODO): Unnecessary details
The MCP server lazily loads the BGE-M3 model on the first request and caches it in memory, ensuring subsequent vector searches are instantaneous.

(TODO): I believe we have more OPs defined as tool, maybe not exposed as MCP.
It would be nice to uniform this pattern, and maybe having an script to automatically regenerate a dedicated md with the entire list.
### Exposed Operations

*   **`list_documents`**: Returns all tax regulatory documents and guidelines currently loaded in the database.
*   **`query_tax_knowledge(query_text: str, limit: int = 5, jurisdiction: Optional[str] = None)`**: Performs high-precision semantic search. It leverages local Apple Silicon GPU acceleration (MPS) to compute query embeddings and executes similarity matching.
*   **`get_chunk_content(chunk_id: int)`**: Retrieves the complete Markdown text, page number, and original document reference for a specific chunk.
*   **`get_surrounding_context(document_name: str, chunk_index: int, before: int = 1, after: int = 1)`**: Fetches neighboring chunks chronologically before and after a specific segment. This enables the querying agent to expand its field of view and read full contexts across page/chunk splits.
*   **`get_document_statistics(document_name: str)`**: Retrieves total page counts and sequential chunk size indicators for a source reference.

### Registering the Server

TODO: Why it reference a local package ? Maybe we should a way to distribute it in a better way.
#### For Claude Desktop or Cursor Configuration
To register this toolset in your AI workspace, add the following configuration block to your MCP config file (e.g., `~/Library/Application Support/Claude/claude_desktop_config.json` or your Cursor settings):

```json
{
  "mcpServers": {
    "tax-compliance-context": {
      "command": "/Users/gio/repos/tax-return-ai/.venv/bin/python3",
      "args": [
        "/Users/gio/repos/tax-return-ai/backend/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/gio/repos/tax-return-ai/backend"
      }
    }
  }
}
```

#### Running it Manually (Standard Input/Output)
To run the stdio JSON-RPC server directly in your terminal:
```bash
source .venv/bin/activate
python backend/mcp_server.py
```
*(Note: All standard logs and diagnostic messages are redirected to `sys.stderr` to keep the stdio JSON-RPC JSON pipes perfectly clean).*
