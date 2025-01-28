"""
Test cases for utils/embeddings/embeddings.py
"""

import pytest
from ..utils.embeddings.embeddings import Embeddings

class TestEmbeddings(Embeddings):
    """Test implementation of the Embeddings interface for testing purposes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

def test_embed_documents():
    """Test embedding documents using the Embeddings interface."""
    embeddings = TestEmbeddings()
    texts = ["text1", "text2"]
    result = embeddings.embed_documents(texts)
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_embed_query():
    """Test embedding a query using the Embeddings interface."""
    embeddings = TestEmbeddings()
    text = "query"
    result = embeddings.embed_query(text)
    assert result == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_aembed_documents():
    """Test asynchronous embedding of documents using the Embeddings interface."""
    embeddings = TestEmbeddings()
    texts = ["text1", "text2"]
    result = await embeddings.aembed_documents(texts)
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_aembed_query():
    """Test asynchronous embedding of a query using the Embeddings interface."""
    embeddings = TestEmbeddings()
    text = "query"
    result = await embeddings.aembed_query(text)
    assert result == [0.1, 0.2, 0.3]
