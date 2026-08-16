import torch


def auto_detect_device() -> str:
    """Auto-detect the best available device for running PyTorch models.

    Returns:
        Device name string ('mps', 'cuda', or 'cpu').
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pool_and_normalize(token_embeddings: torch.Tensor, dim: int = 0) -> list[float]:
    """Mean-pool token embeddings, L2-normalize, move to CPU, and return Python floats list.

    Purpose of Mean Pooling:
        1. Dimension Reduction: Converts a variable-length token sequence tensor of shape
           [num_tokens, hidden_dim] into a fixed single vector of shape [hidden_dim] (1024 for BGE-M3),
           which is required by vector databases.
        2. Semantic Aggregation: Averages contextual representations across all tokens in the sequence,
           capturing the overall sentence/chunk semantics rather than relying on a single token.
        3. Model Convention: BGE-M3 and SBERT models are explicitly trained and fine-tuned for
           mean-pooled representations.

    References / Sources:
        - SBERT (Reimers & Gurevych, 2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
          https://arxiv.org/abs/1908.10084 (Section 3.2 - Pooling Strategies: Mean Pooling)
        - BGE-M3 (Chen et al., 2024): "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings"
          https://arxiv.org/abs/2402.03216

    Args:
        token_embeddings: PyTorch tensor of token embeddings with shape [num_tokens, hidden_dim].
        dim: Dimension over which to perform mean pooling (default: 0).

    Returns:
        List of floats representing the L2-normalized unit vector.
    """
    vector = torch.mean(token_embeddings, dim=dim)
    vector = torch.nn.functional.normalize(vector, p=2, dim=0)
    return vector.cpu().tolist()
