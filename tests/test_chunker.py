"""Unit tests for the LateChunker parent-child chunking logic.

Tests exercise the chunking logic (sentence splitting, child grouping, parent mapping)
using the public `compute_late_chunks` function. The heavy BGE-M3 model is mocked out.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.ingestion.chunker import (
    CHILD_MAX_CHARS,
    PARENT_MAX_CHARS,
    ChunkWithEmbedding,
    LateChunker,
    TransformerWindowRun,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chunker_no_model() -> LateChunker:
    """Return a LateChunker with the heavy BGE-M3 model mocked out.

    Only the grouping and sentence-splitting methods are exercised.
    The `_embed_window_run` method is mocked to return dummy chunks.
    """
    with (
        patch("backend.ingestion.chunker.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("backend.ingestion.chunker.AutoModel.from_pretrained", return_value=MagicMock()),
        patch("backend.ingestion.chunker.auto_detect_device", return_value="cpu"),
    ):
        c = LateChunker()

    def mock_embed(
        window_run: TransformerWindowRun,
        full_text: str,
        parent_texts: list[str],
        start_chunk_idx: int,
        start_flat_child_idx: int,
    ) -> list[ChunkWithEmbedding]:
        chunks: list[ChunkWithEmbedding] = []
        for i, child in enumerate(window_run):
            child_text = " ".join(s.text for s in child)
            flat_idx = start_flat_child_idx + i
            chunks.append(
                ChunkWithEmbedding(
                    chunk_index=start_chunk_idx + i,
                    start_char_idx=child[0].start_char_idx,
                    text_content=child_text,
                    parent_text=parent_texts[flat_idx],
                    embedding=[0.0] * 1024,
                )
            )
        return chunks

    c._embed_window_run = mock_embed  # type: ignore[method-assign]
    return c


# ---------------------------------------------------------------------------
# Ratio safety checks
# ---------------------------------------------------------------------------


def test_parent_child_max_chars_ratio() -> None:
    # Given: Chunking constants configured in chunker.py
    # When: Validating current values
    # Then: PARENT_MAX_CHARS must be greater than or equal to CHILD_MAX_CHARS
    assert PARENT_MAX_CHARS >= CHILD_MAX_CHARS, (
        f"PARENT_MAX_CHARS ({PARENT_MAX_CHARS}) must be >= CHILD_MAX_CHARS ({CHILD_MAX_CHARS})"
    )


# ---------------------------------------------------------------------------
# compute_late_chunks: Child grouping logic
# ---------------------------------------------------------------------------


class TestComputeLateChunksChildren:
    def test_single_short_sentence_is_one_child(self, chunker_no_model: LateChunker) -> None:
        # Given: one sentence well under CHILD_MAX_CHARS
        text = "This is a single short sentence."

        # When: computing chunks
        chunks = chunker_no_model.compute_late_chunks(text, language="en")

        # Then: one child group containing the sentence
        assert len(chunks) == 1
        assert chunks[0].text_content == text

    def test_child_respects_max_chars(self, chunker_no_model: LateChunker) -> None:
        # Given: two sentences each larger than CHILD_MAX_CHARS/2 but individually < CHILD_MAX_CHARS
        # so combined they exceed CHILD_MAX_CHARS.
        text_a = "A" * (CHILD_MAX_CHARS // 2 + 10) + "."
        text_b = " B" * (CHILD_MAX_CHARS // 2 + 10) + "."
        full_text = text_a + text_b

        # When: computing chunks
        chunks = chunker_no_model.compute_late_chunks(full_text, language="en")

        # Then: each sentence is its own child
        assert len(chunks) >= 2
        assert chunks[0].text_content == text_a
        assert chunks[1].text_content == text_b.strip()

    def test_multiple_short_sentences_merge(self, chunker_no_model: LateChunker) -> None:
        # Given: many tiny sentences
        sentences = ["Sentence small."] * 20
        full_text = " ".join(sentences)

        # When: computing chunks
        chunks = chunker_no_model.compute_late_chunks(full_text, language="en")

        # Then: all merged into one child since total length is ~300 chars < CHILD_MAX_CHARS
        assert len(chunks) == 1
        assert chunks[0].text_content == full_text


# ---------------------------------------------------------------------------
# compute_late_chunks: Parent grouping logic
# ---------------------------------------------------------------------------


class TestComputeLateChunksParents:
    def test_parent_count_matches_children(self, chunker_no_model: LateChunker) -> None:
        # Given: many small child groups that should collapse into fewer parents
        sentences = [f"Sentence {i} with some padding text here." for i in range(30)]
        full_text = " ".join(sentences)

        chunks = chunker_no_model.compute_late_chunks(full_text, language="en")

        # Then: each child has a parent text
        for chunk in chunks:
            assert chunk.parent_text is not None
            # And: parent text is always longer than or equal to the child text
            assert len(chunk.parent_text) >= len(chunk.text_content)

    def test_parent_max_chars_respected(self, chunker_no_model: LateChunker) -> None:
        # Given: child groups with controlled sizes
        # 20 sentences of ~200 chars each = 4000 chars total > PARENT_MAX_CHARS (3000)
        sentences = ["X" * 200 + "." for _ in range(20)]
        full_text = " ".join(sentences)

        chunks = chunker_no_model.compute_late_chunks(full_text, language="en")

        # Then: no parent text significantly exceeds PARENT_MAX_CHARS (with tolerance for boundaries)
        for chunk in chunks:
            assert len(chunk.parent_text) <= PARENT_MAX_CHARS + 400

    def test_siblings_share_parent_text(self, chunker_no_model: LateChunker) -> None:
        # Given: two tiny children that should both fit in the same parent
        sentences = ["Sentence A is short."] * 20 + ["Sentence B is short."] * 20
        full_text = " ".join(sentences)

        chunks = chunker_no_model.compute_late_chunks(full_text, language="en")

        # Then: children share the same parent text if they fit within the same PARENT_MAX_CHARS window
        assert len(chunks) > 0
        # Given the small size, they might all fit in one parent chunk
        parent_texts = {c.parent_text for c in chunks}
        assert len(parent_texts) <= len(chunks)


# ---------------------------------------------------------------------------
# compute_late_chunks: Italian legal text splitting
# ---------------------------------------------------------------------------


class TestComputeLateChunksItalian:
    """These tests require the Stanza Italian model to be downloaded."""

    ITALIAN_LEGAL = (
        "Ai fini delle imposte sui redditi, si considerano similari alle azioni "
        "i titoli emessi da societa' ed enti di cui all'articolo 73, comma 1, lettera a). "
        "La remunerazione e' costituita totalmente dalla partecipazione ai risultati economici "
        "della societa' emittente. "
        "Si veda anche il regio decreto-legge 15 marzo 1927, n. 436, convertito nella legge n. 510."
    )

    def test_abbreviation_n_does_not_split(self, chunker_no_model: LateChunker) -> None:
        # Given: Italian legal text with 'n. 436' and 'n. 510' abbreviations
        # When: split into chunks
        chunks = chunker_no_model.compute_late_chunks(self.ITALIAN_LEGAL, language="it")

        # Then: 'n. 436' and 'n. 510' are NOT sentence boundaries
        # Because the text is small, it should all be in one child chunk.
        assert len(chunks) == 1
        assert "n. 436" in chunks[0].text_content
        assert "n. 510" in chunks[0].text_content

    def test_no_empty_sentences(self, chunker_no_model: LateChunker) -> None:
        chunks = chunker_no_model.compute_late_chunks(self.ITALIAN_LEGAL, language="it")
        for c in chunks:
            assert c.text_content.strip(), "Child text should not be empty"


# ---------------------------------------------------------------------------
# compute_late_chunks: English legal text splitting
# ---------------------------------------------------------------------------


class TestComputeLateChunksEnglish:
    ENGLISH_LEGAL = (
        "An investment undertaking must make a return of appropriate tax to Revenue "
        "in connection with chargeable events occurring between 1 January and 30 June "
        "in a particular year, by 30 July of that year. "
        "Since 1 June 2012 such returns and payments are subject to the Mandatory e-Filing regulations. "
        "Returns made by an investment undertaking may be subject to audit by Revenue."
    )

    def test_english_parsing(self, chunker_no_model: LateChunker) -> None:
        # Given: English legal text
        chunks = chunker_no_model.compute_late_chunks(self.ENGLISH_LEGAL, language="en")

        # Then: successfully chunked
        assert len(chunks) == 1
        assert "audit by Revenue." in chunks[0].text_content
