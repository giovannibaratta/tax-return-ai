"""Markdown parser for converting research and advisory documents into parsed pages."""

import os
import re

from backend.ingestion.parser import ParsedPage

# Regex pattern to match HTML tags such as <img ... />, <br>, <p>...</p>, etc.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class MarkdownParser:
    """Parse markdown research documents into ParsedPage objects for chunking."""

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        return _HTML_TAG_RE.sub("", text)

    @classmethod
    def parse_markdown(cls, file_path: str) -> ParsedPage | None:
        """Read a markdown file and return as a single ParsedPage.

        Args:
            file_path: Absolute or relative path to the markdown file.

        Returns:
            ParsedPage with page_number=0, or None if document is empty.

        Raises:
            FileNotFoundError: If file_path does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            raw_content = f.read()

        cleaned_content = cls._strip_html(raw_content).strip()
        if not cleaned_content:
            return None

        return ParsedPage(page_number=0, combined_content=cleaned_content)
