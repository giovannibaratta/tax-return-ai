import math
from unittest.mock import MagicMock, patch

import torch

from backend.llm.embedding_runner import BaseEmbeddingRunner, BgeM3EmbeddingRunner
from backend.utils.utils import pool_and_normalize


def test_pool_and_normalize():
    # Given: A token embeddings tensor with known values on CUDA/MPS or CPU
    tensor = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float32)

    # When: pool_and_normalize is called over dimension 0
    # Mean vector is [1.5, 2.0], norm is sqrt(1.5^2 + 2.0^2) = 2.5
    # Normalized vector is [1.5 / 2.5, 2.0 / 2.5] = [0.6, 0.8]
    res = pool_and_normalize(tensor, dim=0)

    # Then: It returns a list of floats normalized to unit L2 norm
    assert len(res) == 2
    assert math.isclose(res[0], 0.6, rel_tol=1e-5)
    assert math.isclose(res[1], 0.8, rel_tol=1e-5)


@patch("backend.llm.embedding_runner.AutoModel")
@patch("backend.llm.embedding_runner.AutoTokenizer")
def test_bgem3_embedding_runner_embed(mock_tokenizer_cls, mock_model_cls):
    # Given: A mocked AutoTokenizer and AutoModel
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    # Mock tokenizer output dictionary
    mock_inputs = {
        "input_ids": torch.tensor([[101, 2054, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.return_value = mock_inputs

    mock_model = MagicMock()
    mock_model_cls.from_pretrained.return_value.to.return_value.eval.return_value = mock_model

    # Mock model output: last_hidden_state of shape [1, 3, 4]
    mock_output = MagicMock()
    hidden_state = torch.ones((1, 3, 4), dtype=torch.float32)
    mock_output.last_hidden_state = hidden_state
    mock_model.return_value = mock_output

    # When: Initializing BgeM3EmbeddingRunner and generating an embedding for a text
    runner = BgeM3EmbeddingRunner()
    embedding = runner.embed("Test tax return question")

    # Then: The result is a list of floats representing the L2-normalized unit vector
    assert isinstance(runner, BaseEmbeddingRunner)
    assert isinstance(embedding, list)
    assert len(embedding) == 4
    # All tokens are ones, so mean vector is [1, 1, 1, 1], norm is sqrt(4) = 2, normalized is [0.5, 0.5, 0.5, 0.5]
    for val in embedding:
        assert math.isclose(val, 0.5, rel_tol=1e-5)
