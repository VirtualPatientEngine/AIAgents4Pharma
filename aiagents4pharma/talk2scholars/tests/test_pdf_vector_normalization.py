"""Unit tests for vector normalization utilities for GPU COSINE support."""

import logging

import pytest
from langchain_core.embeddings import Embeddings

from aiagents4pharma.talk2scholars.tools.pdf.utils import vector_normalization as vn


def test_normalize_vector_nonzero():
    """Test normalizing a non-zero vector."""
    vec = [3.0, 4.0]
    result = vn.normalize_vector(vec)
    expected = [0.6, 0.8]
    assert pytest.approx(result) == expected


def test_normalize_vector_zero_logs_warning(caplog):
    """Test normalizing a zero vector logs a warning."""
    with caplog.at_level(logging.WARNING):
        result = vn.normalize_vector([0.0, 0.0])
        assert result == [0.0, 0.0]
        assert "Zero vector encountered" in caplog.text


def test_normalize_vectors_batch_empty():
    """Test that an empty batch returns unchanged."""
    result = vn.normalize_vectors_batch([])
    assert result == []


def test_normalize_vectors_batch_normal_case():
    """Test batch normalization of valid vectors with equal dimensions."""
    vectors = [[3, 4], [6, 8]]
    result = vn.normalize_vectors_batch(vectors)
    expected = [
        [0.6, 0.8],
        [0.6, 0.8],
    ]
    for r, e in zip(result, expected, strict=False):
        assert pytest.approx(r) == e


def test_normalize_vectors_batch_with_zero_vector(caplog):
    """Test that zero vectors are handled and logged."""
    vectors = [[0.0, 0.0], [1.0, 0.0]]
    with caplog.at_level(logging.WARNING):
        result = vn.normalize_vectors_batch(vectors)
        assert len(result) == 2
        assert "zero vectors during batch normalization" in caplog.text
        assert pytest.approx(result[1]) == [1.0, 0.0]


class DummyEmbedding(Embeddings):
    """A dummy embedding class for testing normalization wrapper."""

    def __init__(self):
        self.test_attr = "test"

    def embed_documents(self, texts):
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text):
        return [3.0, 4.0]


def test_normalizing_embeddings_embed_documents():
    """Test that document embeddings are normalized."""
    model = vn.NormalizingEmbeddings(DummyEmbedding())
    result = model.embed_documents(["doc1", "doc2"])
    assert len(result) == 2
    assert pytest.approx(result[0]) == [0.6, 0.8]


def test_normalizing_embeddings_embed_query():
    """Test that query embeddings are normalized."""
    model = vn.NormalizingEmbeddings(DummyEmbedding())
    result = model.embed_query("query")
    assert pytest.approx(result) == [0.6, 0.8]


def test_normalizing_embeddings_passthrough():
    """Test attribute delegation to base embedding model."""
    dummy = DummyEmbedding()
    model = vn.NormalizingEmbeddings(dummy)
    assert model.test_attr == "test"


@pytest.mark.parametrize(
    "has_gpu,use_cosine,expected_log",
    [
        (True, True, "ENABLED"),
        (False, True, "DISABLED"),
        (True, False, "DISABLED"),
        (False, False, "DISABLED"),
    ],
)
def test_should_normalize_vectors_logging(has_gpu, use_cosine, expected_log, caplog):
    """Test should_normalize_vectors decision logic and logging."""
    with caplog.at_level(logging.INFO):
        result = vn.should_normalize_vectors(has_gpu, use_cosine)
        if has_gpu and use_cosine:
            assert result is True
        else:
            assert result is False
        assert expected_log in caplog.text


def test_wrap_embedding_model_if_needed_enabled():
    """Test that wrapping is applied when needed."""
    base = DummyEmbedding()
    wrapped = vn.wrap_embedding_model_if_needed(base, has_gpu=True, use_cosine=True)
    assert isinstance(wrapped, vn.NormalizingEmbeddings)


def test_wrap_embedding_model_if_needed_disabled():
    """Test that original model is returned when normalization not needed."""
    base = DummyEmbedding()
    wrapped = vn.wrap_embedding_model_if_needed(base, has_gpu=False, use_cosine=True)
    assert wrapped is base
