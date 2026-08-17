"""Registry of available PDF parsers."""

from __future__ import annotations

from backend.ingestion.ocr_parser import ChandraAPIParser, ChandraOCRParser
from backend.ingestion.parser import BasePDFParser, PDFDocumentParser


def get_parser_registry() -> dict[str, type[BasePDFParser]]:
    """Return dictionary mapping parser names to parser classes."""
    return {
        "pdfplumber": PDFDocumentParser,
        "chandra": ChandraOCRParser,
        "chandra_api": ChandraAPIParser,
    }
