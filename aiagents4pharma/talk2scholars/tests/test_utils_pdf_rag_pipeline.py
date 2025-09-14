"""pdf rag pipeline tests."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline import (
    retrieve_and_rerank_chunks,
)


@pytest.fixture(name="base_config_fixture")
def _base_config_fixture():
    """Provides a config-like object for testing."""
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "initial_retrieval_k": 120,
        "mmr_diversity": 0.7,
    }.get(key, default)
    config.top_k_chunks = 5
    return config


@pytest.fixture(name="mock_docs_fixture")
def _mock_docs_fixture():
    """Simulates PDF document chunks."""
    return [
        Document(page_content=f"chunk {i}", metadata={"paper_id": f"P{i % 2}"}) for i in range(10)
    ]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.rerank_chunks")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.retrieve_relevant_chunks")
def test_rag_pipeline_gpu_path(mock_retrieve, mock_rerank, base_config_fixture, mock_docs_fixture):
    """test RAG pipeline with GPU path."""
    mock_retrieve.return_value = mock_docs_fixture
    mock_rerank.return_value = mock_docs_fixture[:5]

    result = retrieve_and_rerank_chunks(
        vector_store=MagicMock(),
        query="Explain AI.",
        config=base_config_fixture,
        call_id="gpu_test",
        has_gpu=True,
    )

    assert result == mock_docs_fixture[:5]
    mock_retrieve.assert_called_once()
    mock_rerank.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.rerank_chunks")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.retrieve_relevant_chunks")
def test_rag_pipeline_cpu_path(mock_retrieve, mock_rerank, base_config_fixture, mock_docs_fixture):
    """rag pipeline with CPU path."""
    mock_retrieve.return_value = mock_docs_fixture
    mock_rerank.return_value = mock_docs_fixture[:5]

    result = retrieve_and_rerank_chunks(
        vector_store=MagicMock(),
        query="Explain quantum physics.",
        config=base_config_fixture,
        call_id="cpu_test",
        has_gpu=False,
    )

    assert result == mock_docs_fixture[:5]
    mock_retrieve.assert_called_once()
    mock_rerank.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.rerank_chunks")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.retrieve_relevant_chunks")
def test_rag_pipeline_empty_results(mock_retrieve, mock_rerank, base_config_fixture):
    """rag pipeline with no results."""
    mock_retrieve.return_value = []

    result = retrieve_and_rerank_chunks(
        vector_store=MagicMock(),
        query="No match?",
        config=base_config_fixture,
        call_id="empty_test",
        has_gpu=False,
    )

    assert result == []
    mock_rerank.assert_not_called()
