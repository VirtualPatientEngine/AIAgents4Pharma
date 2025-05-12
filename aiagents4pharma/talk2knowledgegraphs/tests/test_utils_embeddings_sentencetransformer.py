"""
Test cases for utils/embeddings/sentence_transformer.py
"""

# Standard library imports
import os

# Third-party imports
import torch
import numpy as np
import pytest

# Local application imports
from ..utils.embeddings.sentence_transformer import EmbeddingWithSentenceTransformer


def get_device():
    """
    Determine the device to use: CPU on CI or if MPS isn't available,
    otherwise MPS.
    """
    if os.getenv("CI") == "true":
        return torch.device("cpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(name="embedding_model")
def embedding_model_fixture():
    """
    Fixture for creating an instance of EmbeddingWithSentenceTransformer.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v1"  # Small model for testing
    embedding_model = EmbeddingWithSentenceTransformer(model_name=model_name)
    # Move underlying model to the correct device before testing
    embedding_model.model.to(get_device())
    return embedding_model


def test_embed_documents(embedding_model):
    """
    Test the embed_documents method of EmbeddingWithSentenceTransformer class.
    """
    # Perform embedding
    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = embedding_model.embed_documents(texts)
    # Check the result
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == 384
    assert embeddings.dtype == np.float32


def test_embed_query(embedding_model):
    """
    Test the embed_query method of EmbeddingWithSentenceTransformer class.
    """
    # Perform embedding
    text = "This is a test query."
    embedding = embedding_model.embed_query(text)
    # Check the result
    assert len(embedding) > 0
    assert len(embedding) == 384
    assert embedding.dtype == np.float32
