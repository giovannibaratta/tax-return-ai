import pytest

from backend.ingestion.helpers import extract_jurisdiction_from_path
from backend.ingestion.ingest import PARSER_REGISTRY, find_page_number
from backend.ingestion.ocr_parser import ChandraAPIParser, ChandraOCRParser
from backend.ingestion.parser import BasePDFParser, PDFDocumentParser


def test_find_page_number_success():
    # Given: Valid page ranges for concatenated batch text
    page_ranges = [
        (0, 100, 1),
        (101, 250, 2),
        (251, 400, 3),
    ]

    # When: Querying character indices within bounds
    page_start = find_page_number(0, page_ranges)
    page_mid = find_page_number(150, page_ranges)
    page_end = find_page_number(400, page_ranges)

    # Then: Correct page numbers are returned
    assert page_start == 1
    assert page_mid == 2
    assert page_end == 3


def test_find_page_number_out_of_bounds_raises():
    # Given: Page ranges ending at offset 200
    page_ranges = [(0, 100, 1), (101, 200, 2)]

    # When / Then: Out of bounds index raises ValueError
    with pytest.raises(ValueError, match="out of bounds"):
        find_page_number(250, page_ranges)


def test_find_page_number_empty_ranges_raises():
    # Given: Empty page ranges
    page_ranges: list[tuple[int, int, int]] = []

    # When / Then: Querying raises ValueError
    with pytest.raises(ValueError, match="page_ranges list is empty"):
        find_page_number(10, page_ranges)


def test_parser_registry_and_interface():
    # Given: PARSER_REGISTRY dictionary
    # When: Checking registered parsers
    # Then: All registered classes inherit from BasePDFParser interface
    for parser_name, parser_cls in PARSER_REGISTRY.items():
        assert issubclass(parser_cls, BasePDFParser)
        assert hasattr(parser_cls, "parse_pdf")


def test_extract_jurisdiction_from_path():
    # Given: Regulation file paths
    path_italy = "data/raw_sources/regulations/italy/law_2024.pdf"
    path_ireland = "data/raw_sources/regulations/ireland/guidance.pdf"
    path_invalid = "somewhere/else/italy_doc.pdf"

    # When: Extracting jurisdiction
    j_italy = extract_jurisdiction_from_path(path_italy)
    j_ireland = extract_jurisdiction_from_path(path_ireland)

    # Then: Correct jurisdiction strings returned
    assert j_italy == "italy"
    assert j_ireland == "ireland"

    # Then: Invalid path structure raises ValueError
    with pytest.raises(ValueError, match="Invalid file path structure"):
        extract_jurisdiction_from_path(path_invalid)
