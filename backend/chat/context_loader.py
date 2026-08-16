"""Utilities for listing and reading document files from the data directory."""

from __future__ import annotations

from pathlib import Path

from backend.chat.models import AttachedContextDoc


def get_data_dir() -> Path:
    """Return absolute path to project data directory.

    Returns:
        Path object pointing to data directory.
    """
    # TODO: Why are we accesing the parent 3 times. This logic is totally unclear. Is it even safe ?
    # I assume because we expect to find the data dir, but maybe this is not the best way to handle it. What if we change the layout ? This will silently (or loudly break in a non good way).
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_context_doc(file_path: str | Path, base_dir: str | Path | None = None) -> AttachedContextDoc:
    """Load context document from disk and read text content verbatim.

    Args:
        file_path: Path to target file (absolute or relative).
        base_dir: Optional base directory for computing relative path.

    Returns:
        AttachedContextDoc dataclass populated with content and metadata.
    """
    path = Path(file_path).resolve()
    base = Path(base_dir).resolve() if base_dir else get_data_dir().resolve()

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
                import fitz  # PyMuPDF

                doc = fitz.open(path)
                pages: list[str] = [
                    str(page.get_text())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    for page in doc
                ]
                content = "\n".join(pages)
            except Exception:
                # TODO: This doesn't seem the good thing to do. We should fail and let the UI know that we cannot read it.
                content = f"[Binary PDF Document: {path.name}]"
        else:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as err:
                # TODO: This doesn't seem the good thing to do. We should fail and let the UI know that we cannot read it.
                content = f"[Error reading file {path.name}: {err}]"

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
