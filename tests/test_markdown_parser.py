"""Unit tests for MarkdownParser."""

from pathlib import Path

import pytest

from backend.ingestion.markdown_parser import MarkdownParser


def test_strip_html():
    # Given: Markdown text with embedded HTML tags
    raw_md = '<img src="https://example.com/logo.png" style="height:64px"/>\n# Title\nSome <p>paragraph</p> with <br/> line breaks.'

    # When: Stripping HTML
    cleaned = MarkdownParser._strip_html(raw_md)

    # Then: HTML tags are removed while markdown text remains
    assert "<img" not in cleaned
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "<br/>" not in cleaned
    assert "# Title" in cleaned
    assert "Some paragraph with  line breaks." in cleaned.replace("\n", " ")


def test_parse_markdown_returns_single_page(tmp_path: Path):
    # Given: A markdown file with multiple sections
    md_content = """# Main Document Title

Introduction text explaining the scope.

## Section 1: Capital Gains Tax
CGT is charged at 33% on disposal of assets.
Annual exemption is €1,270.

## Section 2: Offshore Funds
Offshore funds under Part 27 TCA 1997 are subject to 41% exit tax.
"""
    file_path = tmp_path / "research_test.md"
    file_path.write_text(md_content, encoding="utf-8")

    # When: Parsing markdown
    page = MarkdownParser.parse_markdown(str(file_path))

    # Then: A single ParsedPage is returned containing full document content
    assert page is not None
    assert page.page_number == 0
    assert "Introduction text explaining the scope." in page.combined_content
    assert "## Section 1: Capital Gains Tax" in page.combined_content
    assert "## Section 2: Offshore Funds" in page.combined_content


def test_parse_markdown_empty_file(tmp_path: Path):
    # Given: An empty markdown file or whitespace only
    file_path = tmp_path / "empty.md"
    file_path.write_text("   \n\n  ", encoding="utf-8")

    # When: Parsing empty markdown
    page = MarkdownParser.parse_markdown(str(file_path))

    # Then: Returns None
    assert page is None


def test_parse_markdown_file_not_found():
    # Given: Non-existent file path
    non_existent = "/non/existent/path/doc.md"

    # When / Then: FileNotFoundError is raised
    with pytest.raises(FileNotFoundError, match="Markdown file not found"):
        MarkdownParser.parse_markdown(non_existent)
