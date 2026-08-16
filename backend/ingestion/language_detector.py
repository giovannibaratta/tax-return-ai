"""Language detection helper based on lingua-language-detector.

Supports English and Italian detection for document ingestion and text processing.
"""

import os

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

# Build detector restricted to English and Italian
_DETECTOR: LanguageDetector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.ITALIAN,
).build()


def detect_language(text: str) -> str:
    """Detect whether text is Italian or English using lingua.

    Args:
        text: Raw text string to analyze.

    Returns:
        ISO 639-1 code ('it' or 'en'). Defaults to 'en' if detection is inconclusive.
    """
    if not text or not text.strip():
        return "en"

    detected: Language | None = _DETECTOR.detect_language_of(text)
    if detected == Language.ITALIAN:
        return "it"
    return "en"


def detect_file_language(file_path: str, max_chars: int = 10000) -> str:
    """Detect language of a text or markdown file by inspecting sample content.

    Args:
        file_path: Path to the text or markdown file.
        max_chars: Maximum character count to read from file head for detection.

    Returns:
        ISO 639-1 code ('it' or 'en'). Defaults to 'en' if detection is inconclusive.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for language detection: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        sample = f.read(max_chars)
    return detect_language(sample)
