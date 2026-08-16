"""Cross-Encoder Reranker module using BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from sentence_transformers import CrossEncoder

from backend.utils.utils import auto_detect_device

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

T = TypeVar("T")


@dataclass(frozen=True)
class RerankedResult(Generic[T]):
    """Generic wrapper pairing a candidate item with its computed rerank score.

    Attributes:
        item: Original candidate item object or mapping.
        rerank_score: Cross-Encoder relevance score for the query-item pair.
    """

    item: T
    rerank_score: float


class BgeCrossEncoderReranker:
    """Cross-Encoder reranker backed by BAAI/bge-reranker-v2-m3.

    Cross-Encoder re-ranking is a two-stage retrieval mechanism. In the first stage,
    vector embedding search or keyword search retrieves a broad candidate pool
    (high recall). In the second stage, this reranker processes the query and candidate
    text jointly via full cross-attention transformer layers, producing fine-grained
    relevance scores to re-rank the candidates with higher precision.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANKER_MODEL) -> None:
        """Initialize CrossEncoder model on auto-detected compute device.

        Args:
            model_name: HuggingFace model identifier for the CrossEncoder.
        """
        self.model_name = model_name
        self._device = auto_detect_device()
        logger.info("Initializing BgeCrossEncoderReranker with %s on %s", model_name, self._device)
        self.model = CrossEncoder(model_name, device=self._device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[T],
        text_extractor: Callable[[T], str],
        top_k: int = 5,
    ) -> list[RerankedResult[T]]:
        """Rerank candidate items by cross-attention score with query.

        Re-ranking passes the query string paired with each candidate text through
        a joint cross-attention model to compute a semantic relevance score.
        The top_k highest-scoring candidates are returned as RerankedResult[T]
        instances containing the original candidate item and explicit rerank_score float.

        Args:
            query: The search query string.
            candidates: Sequence of candidate dictionaries or mappings to rerank.
            text_extractor: Callable that extracts the text string from the item.
            top_k: Number of top reranked items to return. Must be > 0.

        Returns:
            Reranked subset of RerankedResult[T] instances ordered by descending relevance score.
        """
        if not candidates or top_k <= 0:
            return []

        pairs: list[tuple[str, str]] = [(query, str(text_extractor(item))) for item in candidates]

        scores = self.model.predict(pairs)  # pyright: ignore[reportUnknownMemberType]

        results: list[RerankedResult[T]] = []
        for score, item in zip(scores, candidates):
            score_float = float(score)  # pyright: ignore[reportArgumentType]
            results.append(RerankedResult(item=item, rerank_score=score_float))

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]


