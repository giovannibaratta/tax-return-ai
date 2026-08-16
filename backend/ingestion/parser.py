from abc import ABC, abstractmethod
from typing import Any, override

import pdfplumber
import pypdf
from pydantic import BaseModel


class ParsedPage(BaseModel):
    page_number: int
    combined_content: str
    layout_blocks: list[dict[str, Any]] | None = None


class BasePDFParser(ABC):
    @classmethod
    @abstractmethod
    def parse_pdf(cls, file_path: str, force_parsing: bool = False, **kwargs: object) -> list[ParsedPage]:
        """Parse PDF file into list of ParsedPage objects."""
        pass


def _format_table_as_markdown(table: list[list[str | None]]) -> str | None:
    """Convert a table (list of list of strings) extracted by pdfplumber into a clean Markdown table."""
    if not table or not any(table):
        return None

    # Clean up cell text: replace None with empty strings, clean newlines & excessive spaces
    cleaned_table: list[list[str]] = []

    for row in table:
        cleaned_row: list[str] = []

        for cell in row:
            val = str(cell or "").replace("\n", " ").strip()
            # Escape pipe symbols to prevent breaking markdown tables
            val = val.replace("|", "\\|")
            cleaned_row.append(val)
        cleaned_table.append(cleaned_row)

    # Skip tables that are entirely empty
    if not any(any(cell for cell in row) for row in cleaned_table):
        return None

    num_cols = len(cleaned_table[0])
    headers = cleaned_table[0]

    for row in cleaned_table:
        # Find row with highest number of columns, so we can pad others
        num_cols = max(num_cols, len(row))

    # Pad rows that are shorter than the num_cols
    for row in cleaned_table:
        if len(row) < num_cols:
            row.extend([""] * (num_cols - len(row)))

    # Column widths for pretty alignment
    col_widths: list[int] = []

    for col_idx in range(num_cols):
        max_w: int = 3  # Minimum width for visual symmetry
        for row in cleaned_table:
            max_w = max(max_w, len(row[col_idx]))
        col_widths.append(max_w)

    markdown_lines: list[str] = []

    # 1. Header row
    header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    markdown_lines.append(header_row)

    # 2. Separator row
    separator_row = "| " + " | ".join("-" * w for w in col_widths) + " |"
    markdown_lines.append(separator_row)

    # 3. Data rows
    for row in cleaned_table[1:]:
        data_row = "| " + " | ".join(f"{c:<{w}}" for c, w in zip(row, col_widths)) + " |"
        markdown_lines.append(data_row)

    return "\n".join(markdown_lines)


class PDFDocumentParser(BasePDFParser):
    @classmethod
    @override
    def parse_pdf(cls, file_path: str, force_parsing: bool = False, **kwargs: object) -> list[ParsedPage]:
        """Parse a PDF document page by page, extracting text and tables.

        Attempt to process the document using pdfplumber, falling back to pypdf for text extraction
        if pdfplumber fails to extract text from a page.

        Returns a list of ParsedPage objects containing page contents.
        """
        parsed_pages: list[ParsedPage] = []

        pypdf_reader = pypdf.PdfReader(file_path)

        print(f"Parsing PDF: {file_path}")

        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1

                # Extract text via pdfplumber
                page_text = page.extract_text()
                page_text = page_text.strip()

                # Fallback to pypdf if pdfplumber failed to locate character elements (e.g. image-based or compressed streams)
                if not page_text:
                    if page_idx >= len(pypdf_reader.pages):
                        raise Exception(f"PDF {file_path} has more pages than pypdf_reader can access.")

                    print(f"  * Page {page_num}: pdfplumber extracted 0 chars. Retrying with pypdf fallback...")
                    page_text = pypdf_reader.pages[page_idx].extract_text()
                    page_text = page_text.strip()

                # Extract tables
                tables = page.extract_tables()
                markdown_tables: list[str] = []

                for table in tables:
                    md_table = _format_table_as_markdown(table)
                    if md_table:
                        markdown_tables.append(md_table)

                # Combine standard text with formatted tables
                combined_content = page_text

                if markdown_tables:
                    combined_content += "\n\n### Extracted Tables:\n" + "\n\n".join(markdown_tables)

                combined_content = combined_content.strip()

                if not combined_content:
                    raise Exception(
                        f"PDF {file_path} Page {page_num} has 0 characters extracted by both pdfplumber and pypdf."
                    )

                parsed_pages.append(ParsedPage(page_number=page_num, combined_content=combined_content.strip()))

        print(f"Finished parsing. Extracted {len(parsed_pages)} pages.")
        return parsed_pages


def get_parser_registry() -> dict[str, type[BasePDFParser]]:
    """Return dictionary mapping parser names to parser classes.

    TODO: if we have circular import, maybe we are structuring the code poorly?
    Lazy imports Chandra parser classes to prevent circular imports.
    """
    from backend.ingestion.ocr_parser import ChandraAPIParser, ChandraOCRParser

    return {
        "pdfplumber": PDFDocumentParser,
        "chandra": ChandraOCRParser,
        "chandra_api": ChandraAPIParser,
    }
