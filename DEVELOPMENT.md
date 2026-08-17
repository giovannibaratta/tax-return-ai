# Development Guide

This document outlines development practices, environment setup, database migrations, and testing guidelines.

---

## 🛠️ Environment Setup

Before running tests or scripts, set up your Python virtual environment:

### 1. Set Up the Virtual Environment
Create and activate your Python virtual environment inside the project root:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Tests

The test suite is built on top of the `pytest` framework. 

### 1. Run All Tests
```bash
# With active virtual environment
pytest

# Without activating virtual environment
.venv/bin/pytest
```

### 2. Run Specific Test Files
```bash
pytest tests/test_financial_records.py
```

### 3. Run Specific Test Cases
```bash
pytest tests/test_financial_records.py::test_strict_financial_record_validation
```

### 4. Run with Verbose / Print Output
```bash
pytest -v -s
```

---

## 📐 Test Coding Guidelines

Please adhere to the coding rules in [AGENTS.md](AGENTS.md) when writing new tests:

1. **Use Pytest Framework**: Write and run tests using `pytest` framework instead of `unittest`.
2. **Follow the Structure Pattern**: Write tests using the `Given / When / Then` documentation pattern:
   ```python
   def test_example_behavior():
       # Given: <context or setup>
       input_value = "RSSMRC80A01F205Z"

       # When: <action or function call>
       result = validate_codice_fiscale(input_value)

       # Then: <expected outcome or assertions>
       assert result["valid"] is True
   ```
3. **Database Test Isolation**: If a test interacts with a database, ensure it operates on an in-memory database (`MemoryDb()`) or temporary database file, as demonstrated across [tests/test_financial_records.py](tests/test_financial_records.py).

---

## 🗄️ Database Migrations

This project uses **Liquibase** with YAML changelogs to manage database schema migrations.

### Prerequisites

You must have the Java Runtime Environment (JRE) and the Liquibase CLI installed on your machine. On macOS:
```bash
brew install openjdk liquibase
```

### Migration File Structure

Migrations are organized inside the `db-migrations/` directory starting with `db-migrations/root-changelog.yaml`. Each version has its own subdirectory (e.g. `db-migrations/v1/`) containing self-documenting YAML change sets.

### Running Migrations

#### 1. Programmatically (Automatic)
Migrations run automatically on application startup whenever the `DatabaseManager` is instantiated (e.g., when running tests or starting the ingestion pipelines).

#### 2. Manually (CLI)
You can run migrations manually from the root directory of the project:
```bash
liquibase --changelog-file=db-migrations/root-changelog.yaml --url=jdbc:sqlite:database/tax_data.db update
```
