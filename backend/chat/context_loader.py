"""Utilities for listing and reading document files from the data directory."""

from __future__ import annotations

from pathlib import Path

import pymupdf as pdf

from backend.chat.models import AttachedContextDoc
from backend.config import get_app_config


def load_context_doc(file_path: str | Path, base_dir: str | Path | None = None) -> AttachedContextDoc:
    """Load context document from disk and read text content verbatim.

    Args:
        file_path: Path to target file (absolute or relative).
        base_dir: Optional base directory for computing relative path.

    Returns:
        AttachedContextDoc dataclass populated with content and metadata.
    """
    path = Path(file_path).resolve()
    base = Path(base_dir).resolve() if base_dir else get_app_config().data_dir.resolve()

    # TODO: What is that ? It looks very messy. Should also this be part of the backend ? Isn't this better suited in the UI package ?
    try:
        rel_path = str(path.relative_to(base))
    except ValueError:
        try:
            rel_path = str(path.relative_to(base.parent))
        except ValueError:
            rel_path = path.name

    content = ""
    if path.exists() and path.is_file():
        if path.suffix.lower() == ".pdf":
            try:
                doc = pdf.open(path)
                pages: list[str] = [
                    str(page.get_text())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    for page in doc
                ]
                content = "\n".join(pages)
            except Exception as err:
                raise ValueError(f"Failed to read PDF document {path.name}: {err}") from err
        else:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as err:
                raise ValueError(f"Failed to read file {path.name}: {err}") from err

    return AttachedContextDoc(
        relative_path=rel_path,
        full_path=str(path),
        content=content,
        char_count=len(content),
    )


def format_prompt_with_context(
    user_prompt: str,
    attached_docs: list[AttachedContextDoc],
) -> str:
    """Assemble final prompt sent to LLM by appending attached documents verbatim.

    Args:
        user_prompt: Original user input prompt text.
        attached_docs: List of AttachedContextDoc instances.

    Returns:
        Formatted prompt string.
    """
    if not attached_docs:
        return user_prompt

    sections = [user_prompt.strip(), "\n\n--- Attached Context Documents ---"]
    for doc in attached_docs:
        sections.append(f"\n\n[Document: {doc.relative_path}]\n{doc.content}")

    return "".join(sections)
