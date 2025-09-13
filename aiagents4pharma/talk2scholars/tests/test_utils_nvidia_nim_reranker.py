"""
Unit tests for NVIDIA NIM reranker error handling in nvidia_nim_reranker.py
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from aiagents4pharma.talk2scholars.tools.pdf.utils import nvidia_nim_reranker
from aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker import (
    rerank_chunks,
)


@pytest.fixture(name="chunks_fixture")
def fixture_chunks():
    """chunks_fixture fixture to simulate PDF chunks."""
    return [
        Document(
            page_content=f"chunk {i}",
            metadata={"paper_id": f"P{i % 2}", "relevance_score": 0.9 - 0.01 * i},
        )
        for i in range(10)
    ]


def test_rerank_chunks_short_input(chunks_fixture):
    """rerank_chunks with fewer chunks than top_k should return original."""
    result = rerank_chunks(chunks_fixture[:3], "What is cancer?", config=MagicMock(), top_k=5)
    assert result == chunks_fixture[:3]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.logger")
def test_rerank_chunks_missing_api_key_logs_and_raises(mock_logger, chunks_fixture):
    """
    If config.reranker.api_key is None:
      - logger.error(...) should be called
      - rerank_chunks should raise ValueError
    """
    mock_config = MagicMock()
    mock_config.reranker.api_key = None

    with pytest.raises(
        ValueError,
        match="Configuration 'reranker.api_key' must be set for reranking",
    ):
        rerank_chunks(chunks_fixture, "What is cancer?", config=mock_config, top_k=5)

    mock_logger.error.assert_called_once_with(
        "No NVIDIA API key found in configuration for reranking"
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank")
def test_rerank_chunks_success(mock_reranker_cls, chunks_fixture):
    """rerank_chunks with successful reranking."""
    reranker_instance = MagicMock()
    reranker_instance.compress_documents.return_value = list(reversed(chunks_fixture))
    mock_reranker_cls.return_value = reranker_instance

    mock_config = MagicMock()
    mock_config.reranker.api_key = "test_key"
    mock_config.reranker.model = "test_model"

    result = rerank_chunks(chunks_fixture, "Explain mitochondria.", config=mock_config, top_k=5)

    assert isinstance(result, list)
    assert result == list(reversed(chunks_fixture))[:5]
    reranker_instance.compress_documents.assert_called_once_with(
        query="Explain mitochondria.", documents=chunks_fixture
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank")
def test_rerank_chunks_reranker_fails_raises_and_calls_compress(mock_reranker_cls, chunks_fixture):
    """
    If NVIDIARerank.compress_documents raises RuntimeError:
      - rerank_chunks should propagate the RuntimeError
      - and compress_documents should have been called
    """
    reranker_instance = MagicMock()
    reranker_instance.compress_documents.side_effect = RuntimeError("API failure")
    mock_reranker_cls.return_value = reranker_instance

    mock_config = MagicMock()
    mock_config.reranker.api_key = "valid_key"
    mock_config.reranker.model = "reranker"

    with pytest.raises(RuntimeError, match="API failure"):
        rerank_chunks(chunks_fixture, "How does light affect plants?", config=mock_config, top_k=3)

    reranker_instance.compress_documents.assert_called_once_with(
        query="How does light affect plants?", documents=chunks_fixture
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.logger")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank")
def test_rerank_chunks_debug_block_triggered(mock_reranker_cls, mock_logger, chunks_fixture):
    """rerank_chunks should log debug info if debug logging is enabled."""
    mock_logger.isEnabledFor.return_value = True

    reranker_instance = MagicMock()
    reranker_instance.compress_documents.return_value = chunks_fixture
    mock_reranker_cls.return_value = reranker_instance

    mock_config = MagicMock()
    mock_config.reranker.api_key = "abc"
    mock_config.reranker.model = "mymodel"

    result = nvidia_nim_reranker.rerank_chunks(
        chunks_fixture * 2, "Test query", mock_config, top_k=3
    )

    assert result == chunks_fixture[:3]
    assert mock_logger.debug.called
