# Python Code Quality Rules

To maintain high readability, maintainability, and explainability across the repository, all agents must follow these rules:

## 1. Strong Typing & Linting
- **Always Lint & Type-Check**: All code changes must be verified using `pyright` / `basedpyright`.
- **Forbid `Any`**: Do not use `Any` or untyped structures (`dict[str, Any]`). Use concrete Pydantic models, SQLModels, dataclasses, precise Unions, or `object` with type checks (`isinstance`).
- **Strict Error Suppression**: Type checker warnings/errors (`# type: ignore`, `# pyright: ignore`) must be suppressed **ONLY** when there is no clean alternative way to resolve them.
- **Prefer Strict Models Over DB Entities**: Avoid passing loose database models with optional/nullable fields to business logic when a stricter, well-validated domain model can be used. Do not bypass model validation or use DB entities directly as a micro-optimization; prioritize type safety, strict invariants, and code clarity.
- **Forbid Masking Fallbacks**: Do not fabricate dummy default values (e.g., `or "unknown"`, `or Decimal("0")`, `or 0`) when validating domain models or records with missing mandatory fields. Raise a `ValueError` or surface validation errors explicitly instead of silently masking invalid records.
- Enforce explicit type hints on all function arguments and return values.

## 2. Explainability & Documentation
- **Public Functions, Methods & Classes**:
  - All public classes, functions, and methods must include standard **Google-style docstrings**.
  - Must include concise summary line, detailed description if necessary, `Args:`, `Returns:`, and `Raises:` sections where applicable.
- **Private Functions (`_foo`)**:
  - Optional for simple, self-explanatory helper functions.
  - **Mandatory docstrings** when function logic becomes complex or input/output structures are unclear (e.g., nested dictionaries, tuples, lists with complex elements).
- **Function Decomposition**:
  - Break down complex, monolithic functions into smaller, single-responsibility helper functions.
- Decouple LLM prompts/schemas from core control logic. Keep mock data generators strictly isolated in test utilities.

## 3. Configuration Management
- Keep API credentials, endpoints, and database paths out of code.
- Manage configuration variables via environment variables or a local `.env` file, loaded explicitly using `python-dotenv`.

## 4. Testing
- Write and run tests using `pytest` framework instead of `unittest`.
- Write tests using the pattern
    ```python
    # Given: <context>
    # When: <action>
    # Then: <expected outcome>
    ```

## 5. Italian Number Parsing Standardization
- **Standard Format**: Use `,` (comma) strictly for the decimal/cents part and NOTHING or `.` (dot) for thousands ("migliaia").
- **Parsing Rule**:
  - `1.091` or `1.091,00` represents integer `1091` (one thousand ninety-one).
  (TODO): Isn't 1091.50 what we don't want ?
  - `1091,50` or `1.091,50` represents `1091.50`.
  - In OCR extractors, parsers, and LLM voter prompts, strip thousands separators (`.`) and convert comma (`,`) to Python/JSON standard decimal dot (`.`).

## 6. Defensive Financial & Critical Computation Post-Conditions
- **Fail Fast on Invalid Outputs**: In financial computations or critical business logic (lot matching, tax calculations, gain/cost basis aggregations, remittance capping, ...), perform explicit post-condition validations on computed values before returning to callers.
- **Raise Instantly**: If a computed value breaks strict domain invariants (e.g., negative cost basis, negative quantity, tax rate outside `0 <= rate <= 1`, matched quantity exceeding sell quantity, or negative total proceeds for positive unit price), raise a `PostconditionError` (leverage `assert_postcondition` function) immediately to abort the flow rather than propagating invalid state.
- **Testing Exemption**: These defensive post-condition checks act as runtime guards against regression and do not require explicit unit tests.

## 7. Data Immutability & Safe Data Copying
- **Immutability by Default**: Treat input arguments, domain models, lists, and records as immutable by default. Never mutate caller-provided data structures or objects in place unless strictly required.
- **Copy Before Return**: If a function modifies or transforms incoming data structures, create and return new objects or shallow/deep copies (e.g., using `model_copy()`, new instances, or list comprehensions) rather than mutating input objects in place.

## 8. Virtual env

The package contains a virtual environment .venv that can be used to execute the python tools for testing, debugging and code quality checks.

- Do not use poetry