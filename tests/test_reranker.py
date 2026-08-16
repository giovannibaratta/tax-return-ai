"""Tests for BgeCrossEncoderReranker and two-stage search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.llm.reranker import BgeCrossEncoderReranker, RerankedResult


@patch("backend.llm.reranker.CrossEncoder")
def test_bge_cross_encoder_reranker(mock_cross_encoder_cls: MagicMock) -> None:
    # Given: A mocked CrossEncoder returning known scores for candidate pairs
    mock_encoder = MagicMock()
    mock_cross_encoder_cls.return_value = mock_encoder
    mock_encoder.predict.return_value = [0.1, 0.95, 0.3]

    candidates: list[dict[str, object]] = [
        {
            "chunk_id": 1,
            "text_content": "Irrelevant form field 8 numeric date of disposal",
            "document_name": "form11.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": 2,
            "text_content": "Section 4.1.5 Calculating an 8 year deemed disposal tax rule for offshore funds.",
            "document_name": "27-04-01.pdf",
            "page_number": 6,
        },
        {
            "chunk_id": 3,
            "text_content": "General introduction to tax return procedures",
            "document_name": "intro.pdf",
            "page_number": 2,
        },
    ]

    reranker = BgeCrossEncoderReranker()
    query = "Calculating an 8 year deemed disposal"

    # When: Reranking candidate items
    reranked = reranker.rerank(
        query=query, candidates=candidates, text_extractor=lambda x: str(x.get("text_content", "")), top_k=2
    )

    # Then: The exact section match is ranked #1 with high score
    assert len(reranked) == 2
    assert isinstance(reranked[0], RerankedResult)
    assert reranked[0].item["chunk_id"] == 2
    assert "27-04-01.pdf" in str(reranked[0].item["document_name"])
    assert reranked[0].rerank_score == 0.95
