import hashlib
import logging
import os

from pydantic import BaseModel


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to only keep alphanumeric characters, dots, underscores, and hyphens."""
    return "".join(c for c in filename if c.isalnum() or c in "._-").strip()


def calculate_sha256(file_path: str) -> str:
    """Calculate the SHA256 hash of a file's content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class IngestionDocument(BaseModel):
    file_path: str
    doc_name: str
    sha: str
    provider: str | None = None
    account_country: str | None = None

    @classmethod
    def from_file(
        cls,
        file_path: str,
        account_country: str | None = None,
        provider: str | None = None,
    ) -> "IngestionDocument":
        doc_name = os.path.basename(file_path)
        sha = calculate_sha256(file_path)
        return cls(
            file_path=file_path,
            doc_name=doc_name,
            sha=sha,
            provider=provider,
            account_country=account_country,
        )

    @property
    def safe_doc_name(self) -> str:
        """Sanitized version of the document name safe for local filenames."""
        return sanitize_filename(self.doc_name)


def extract_jurisdiction_from_path(file_path: str) -> str:
    """Extract jurisdiction from a regulation file path.

    Expected structure: .../raw_sources/regulations/<jurisdiction>/<filename>

    Raises:
        ValueError: If the file path does not contain the expected structure.
    """
    normalized = file_path.replace("\\", "/")
    parts = [p.strip() for p in normalized.split("/") if p.strip()]

    for i in range(len(parts) - 2):
        if parts[i] == "raw_sources" and parts[i + 1] == "regulations":
            return parts[i + 2].lower()

    raise ValueError(
        f"Invalid file path structure. Expected 'raw_sources/regulations/<jurisdiction>' segment in: {file_path}"
    )


def extract_research_jurisdiction_from_path(file_path: str) -> str | None:
    """Extract optional jurisdiction from a research file path.

    Expected structures:
      - .../research/<jurisdiction>/<filename> -> returns <jurisdiction>
      - .../research/<filename> -> returns None

    Args:
        file_path: Path to the research document.

    Returns:
        Jurisdiction name in lower case, or None if located at root research level.
    """
    normalized = file_path.replace("\\", "/")
    parts = [p.strip() for p in normalized.split("/") if p.strip()]

    for i in range(len(parts) - 1):
        if parts[i] == "research":
            # If there is at least one intermediate directory between 'research' and filename
            if i + 2 < len(parts):
                jurisdiction = parts[i + 1].lower()
                return jurisdiction if jurisdiction in {"italy", "ireland"} else None
            return None
    return None


def extract_account_country_and_provider(file_path: str) -> tuple[str, str]:
    """
    Extract account country and provider from a file path.
    Expected structure: .../raw_sources/records/<account_country>/<provider>/<filename>
    Returns (account_country, provider) if valid

    Raise:
        ValueError: If the file path does not contain the expected structure.
    """
    normalized = file_path.replace("\\", "/")
    parts = [p.strip() for p in normalized.split("/") if p.strip()]

    # Check if path has at least 4 parts (raw_sources, records, account_country, provider)
    if len(parts) < 4:
        raise ValueError(f"Invalid file path structure. Expected at least 4 parts, got {len(parts)}: {file_path}")

    # Find the occurrence of 'records' preceded by 'raw_sources'
    idx = -1
    for i in range(len(parts) - 3):
        if parts[i] == "raw_sources" and parts[i + 1] == "records":
            idx = i
            break

    if idx == -1:
        raise ValueError(f"Invalid file path structure. Expected 'raw_sources/records' segment in: {file_path}")

    account_country = parts[idx + 2].lower()
    provider = parts[idx + 3].lower()

    return account_country, provider


def log_env_vars(logger: logging.Logger, level: int = logging.DEBUG):
    """Log all environment variables."""
    for key, value in os.environ.items():
        logger.log(level, f"{key}: {value}")
