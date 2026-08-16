# Development Guide

This document outlines the development practices, environment setup, and instructions on how to run tests in this repository.

## 🛠️ Environment Setup

Before running tests, ensure you have set up your Python virtual environment and installed the dependencies.

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
To run all tests in the repository:
```bash
# With active virtual environment
pytest

# Without activating virtual environment
.venv/bin/pytest
```

### 2. Run Specific Test Files
To run only a specific test suite:
```bash
pytest tests/test_validators.py
```

### 3. Run a Specific Test Case
To run a single test function or test case inside a test class:
```bash
pytest tests/test_validators.py::TestItalyValidators::test_valid_codice_fiscale_male
```

### 4. Run Tests with Verbose Output
For more verbose output showing individual test execution:
```bash
pytest -v
```

### 5. Running Tests with Print/Standard Output Enabled
By default, `pytest` captures standard output. If you want to see `print()` statements during execution:
```bash
pytest -s
```

---

(TODO): Just define the test coding guidelines in a single file, either AGENTS.md or docs/dev/testing uidelines.
## 📐 Test Coding Guidelines


(TODO): Adjust path to be relative to repo root
Please adhere to the following coding rules from [AGENTS.md](file:///Users/gio/repos/tax-return-ai/AGENTS.md) when writing new tests:

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

(TODO): Broken link
3. **Database Test Isolation**: If a test interacts with a database, ensure it operates on a temporary test database (e.g. `database/test_temp.db`) and cleans up the database file after execution, as seen in [tests/test_ingestion_pipeline.py](file:///tests/test_ingestion_pipeline.py).

---

## 🗄️ Database Migrations

This project uses **Liquibase** with YAML changelogs to manage database schema migrations.

### Prerequisites

You must have the Java Runtime Environment (JRE) and the Liquibase CLI installed on your machine. On macOS, install them via Homebrew:
```bash
brew install openjdk liquibase
```

### Migration File Structure
(TODO): No need to mention the individual files, also because they drifted. Yaml should be self-documenting, or having document in place.

Migrations are organized inside the `db-migrations/` directory:
- `db-migrations/root-changelog.yaml`: The root migration index.
- `db-migrations/v1/root-v1.yaml`: Version 1 migration index.
- `db-migrations/v1/20260712223000-create-initial-tables.yaml`: Creates initial schemas.
- `db-migrations/v1/20260712224500-make-fees-and-fx_rate-nullable.yaml`: SQLite drop-and-recreate migration for nullable columns.

### Running Migrations

#### 1. Programmatically (Automatic)
Migrations run automatically on application startup whenever the `DatabaseManager` is instantiated (e.g., when running tests or starting the ingestion pipelines).

#### 2. Manually (CLI)
You can run migrations manually from the root directory of the project:
```bash
liquibase --changelog-file=db-migrations/root-changelog.yaml --url=jdbc:sqlite:database/tax_data.db update
```

