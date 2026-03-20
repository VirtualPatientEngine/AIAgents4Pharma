"""
Test cases for utils/embeddings/sentence_transformer.py
"""

from __future__ import annotations

import numpy as np

from ..utils.embeddings import sentence_transformer as st_module
from ..utils.embeddings.sentence_transformer import EmbeddingWithSentenceTransformer


class FakeSentenceTransformer:
    """Fake sentence transformer returning deterministic arrays."""

    def __init__(self, *_args, **_kwargs):
        """Initialize the fake transformer."""
        self.ready = True

    def to(self, _device):
        """Return self for device placement chaining."""
        return self

    def encode(self, texts, **_kwargs):
        """Return deterministic embeddings for the given texts."""
        if isinstance(texts, list):
            return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_embed_documents(monkeypatch):
    """Test the embed_documents method of EmbeddingWithSentenceTransformer class."""
    monkeypatch.setattr(st_module, "SentenceTransformer", FakeSentenceTransformer, raising=True)
    embedding_model = EmbeddingWithSentenceTransformer(
        model_name="sentence-transformers/all-MiniLM-L6-v1"
    )
    embedding_model.model.to("cpu")

    texts = ["This is a test sentence.", "Another test sentence."]
    embeddings = embedding_model.embed_documents(texts)
    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32


def test_embed_query(monkeypatch):
    """Test the embed_query method of EmbeddingWithSentenceTransformer class."""
    monkeypatch.setattr(st_module, "SentenceTransformer", FakeSentenceTransformer, raising=True)
    embedding_model = EmbeddingWithSentenceTransformer(
        model_name="sentence-transformers/all-MiniLM-L6-v1"
    )

    embedding = embedding_model.embed_query("This is a test query.")
    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
