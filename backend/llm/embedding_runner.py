import logging
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from backend.utils.utils import auto_detect_device, pool_and_normalize

logger = logging.getLogger(__name__)


class BaseEmbeddingRunner(ABC):
    """Abstract base class for embedding generation."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A normalized float list representing the embedding vector.
            Dimension depends on the underlying model (1024 for BGE-M3).
        """
        pass


class BgeM3EmbeddingRunner(BaseEmbeddingRunner):
    """Local embedding runner backed by the BAAI/bge-m3 model via Hugging Face Transformers.

    Uses the same model and normalization strategy as the ingestion pipeline
    (see backend/ingestion/chunker.py), ensuring that query embeddings and
    document chunk embeddings are produced in the same vector space.

    The model is loaded once at construction time and reused across calls.
    Callers should instantiate this class once and inject it as a dependency
    to avoid repeated expensive model loading.

    Attributes:
        model_name: Hugging Face model identifier (default: 'BAAI/bge-m3').
        device: Torch device string ('mps', 'cuda', or 'cpu'), auto-detected if not set.
    """

    def __init__(self) -> None:
        """Initialize the BGE-M3 embedding runner."""
        self.model_name: str = "BAAI/bge-m3"
        self._device: str = auto_detect_device()

        logger.info(
            "Initializing BgeM3EmbeddingRunner with model '%s' on device '%s'",
            self.model_name,
            self._device,
        )
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = (
            AutoModel.from_pretrained(self.model_name).to(self._device).eval()  # type: ignore[assignment]
        )

    def embed(self, text: str) -> list[float]:
        """Generate a normalized 1024-dimensional BGE-M3 embedding for the given text.

        Tokenizes the input, runs a forward pass through BGE-M3, applies mean pooling
        over token embeddings to combine individual token representations into a sequence-level
        vector, then L2-normalizes the result so that dot-product equals cosine similarity.
        Transfers the resulting tensor from accelerator (GPU/MPS) memory to CPU RAM for standard Python list output.

        Args:
            text: The input text to embed (e.g. a space-joined list of keywords).

        Returns:
            A list of 1024 floats representing the normalized embedding vector.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Shape: [1, num_tokens, hidden_dim] -> [num_tokens, hidden_dim]
            token_embeddings = outputs.last_hidden_state[0]

        # Mean-pool over token dimension, L2-normalize, and move to CPU list
        return pool_and_normalize(token_embeddings, dim=0)
