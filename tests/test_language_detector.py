"""Unit tests for language detection helper based on lingua."""

from pathlib import Path

import pytest

from backend.ingestion.language_detector import detect_file_language, detect_language


def test_detect_language_italian():
    # Given: Italian tax law text
    italian_text = "L'imposta sulle plusvalenze finanziarie si applica con aliquota del 26 per cento."

    # When: Detecting language
    lang = detect_language(italian_text)

    # Then: Detected as Italian 'it'
    assert lang == "it"


def test_detect_language_english():
    # Given: English capital gains tax text
    english_text = "Capital Acquisitions Tax and Capital Gains Tax are charged at the statutory rate of 33%."

    # When: Detecting language
    lang = detect_language(english_text)

    # Then: Detected as English 'en'
    assert lang == "en"


def test_detect_language_empty_fallback():
    # Given: Empty and whitespace strings
    empty_str = ""
    whitespace_str = "   \n\t  "

    # When: Detecting language
    lang_empty = detect_language(empty_str)
    lang_ws = detect_language(whitespace_str)

    # Then: Fallback to 'en'
    assert lang_empty == "en"
    assert lang_ws == "en"


def test_detect_file_language(tmp_path: Path):
    # Given: Sample Italian and English markdown files
    it_file = tmp_path / "research_italy.md"
    it_file.write_text(
        "## Quadro RT\nGuida al calcolo delle imposte sostitutive sui redditi diversi.",
        encoding="utf-8",
    )

    en_file = tmp_path / "research_ireland.md"
    en_file.write_text(
        "## Section 1: CGT Rates\nStatutory CGT rate is 33 percent under Section 28 TCA 1997.",
        encoding="utf-8",
    )

    nonexistent_file = tmp_path / "missing.md"

    # When: Detecting file languages
    lang_it = detect_file_language(str(it_file))
    lang_en = detect_file_language(str(en_file))

    # Then: Correct languages identified
    assert lang_it == "it"
    assert lang_en == "en"

    # Then: Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        detect_file_language(str(nonexistent_file))
