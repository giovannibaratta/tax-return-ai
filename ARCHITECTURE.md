# 🏗️ Tax-Return-AI Architecture

This document describes the high-level architecture, pipelines, data structures, and tool layer for the Tax-Return-AI platform.

---

## 1. Regulatory Documents Ingestion Pipeline

Ingests regulatory PDFs and manuals (e.g. Italian TUIR, Irish Revenue manuals) into vector embeddings with Late Chunking.

```mermaid
graph TD
    A["Raw Regulatory Sources<br/>(data/raw_sources/{italy,ireland})"] --> B["PDF Parser<br/>(Chandra OCR / pdfplumber)"]
    B --> C["Language & Boundary Detection<br/>(Stanza NLP it/en)"]
    C --> D["Late Chunking & Token Pooling<br/>(Local BGE-M3 on MPS/CUDA)"]
    D --> E[("SQLite Vector DB<br/>(database/tax_data.db)<br/>- tax_document_metadata<br/>- vss_tax_chunks (sqlite-vec)")]
```

---

## 2. Financial & Income Records Ingestion Pipeline

Ingests financial broker statements (Directa, Interactive Brokers) and official tax income summaries (Irish Revenue EDS, Italian CU) with automated PII sanitization and multi-voter LLM consensus.

```mermaid
graph TD
    F["Financial & Income Documents<br/>(Statements, EDS, Tax Forms)"] --> G["Layout-Aware OCR<br/>(Chandra OCR / Datalab API)"]
    G --> H["PII Sanitization Pipeline<br/>1. LLM Redaction<br/>2. Presidio NER<br/>3. OpenAI Privacy Filter"]
    H --> I["Multi-Voter LLM Extraction<br/>(3 Voter Models in Parallel)"]
    I --> J{"Consensus Engine"}
    J -- "Unanimous / Reconciled" --> K["De-anonymization"]
    K --> L[("Persist to SQLite<br/>- financial_records<br/>- tax_income_records<br/>Status: APPROVED")]
    J -- "Irreconcilable Mismatch" --> M[("Persist Staged / Escalated<br/>Status: ESCALATED_TO_USER")]
    M --> N["UI Resolution & Manual Review<br/>(PySide6 Desktop Application)"]
```

---

## 3. Tool, MCP, & Agent Deliberation Layer

Provides a unified tool registry shared across PydanticAI chat agents, courtroom deliberation agents, and MCP clients.

```mermaid
graph TD
    subgraph STL["Shared Tool Layer"]
        T1["query_tax_knowledge"]
        T2["keyword_search_knowledge"]
        T3["get_chunk_content"]
        T4["get_surrounding_context"]
        T5["get_document_statistics"]
        T6["filter_financial_records"]
        T7["get_tax_income_records"]
    end

    STL --> MCP["MCP Server (FastMCP / stdio)"]
    STL --> Chat["Chat Assistant (PydanticAI)"]
    STL --> Court["Courtroom Deliberation (PydanticAI)"]
```
