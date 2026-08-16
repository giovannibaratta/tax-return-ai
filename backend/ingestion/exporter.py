"""Export OCR processed document pages to filesystem markdown directory structure."""

import re
from pathlib import Path

from backend.domain_models import DocumentPageInfo
from backend.ingestion.ocr_parser import clean_ocr_markdown


def export_pages_to_disk(
    pages: list[DocumentPageInfo],
    target_dir: str = "data/processed",
) -> int:
    """Export a list of DocumentPageInfo objects to filesystem directory hierarchy.

    Writes files to: {target_dir}/{jurisdiction}/{document_name}/page_{page_number}.md

    Args:
        pages: List of DocumentPageInfo domain objects.
        target_dir: Output root directory path.

    Returns:
        Total number of page files successfully written to disk.
    """
    exported_count = 0
    target_path = Path(target_dir)

    for page_data in pages:
        safe_doc_name = re.sub(r"[^\w\.-]", "_", page_data.document_name)
        safe_jur = (page_data.jurisdiction or "general").lower()
        doc_folder = target_path / safe_jur / safe_doc_name
        doc_folder.mkdir(parents=True, exist_ok=True)

        page_text = clean_ocr_markdown(page_data.text_content)
        page_file = doc_folder / f"page_{page_data.page_number}.md"
        header = f"# {page_data.document_name} — Page {page_data.page_number}/{page_data.total_pages} ({page_data.jurisdiction})\n\n"
        page_file.write_text(header + page_text, encoding="utf-8")
        exported_count += 1

    return exported_count
