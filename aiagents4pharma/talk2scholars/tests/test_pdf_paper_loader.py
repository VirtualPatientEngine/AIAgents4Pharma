"""paper_loader tests for the load_all_papers function."""

from unittest.mock import MagicMock, patch

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader import (
    load_all_papers,
)


@pytest.fixture
def articles():
    """A fixture to provide a sample articles dictionary."""
    return {
        "p1": {"pdf_url": "http://example.com/p1.pdf", "title": "Paper 1"},
        "p2": {"pdf_url": "http://example.com/p2.pdf", "title": "Paper 2"},
        "p3": {"title": "No PDF paper"},
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store fixture."""
    return MagicMock(
        loaded_papers={"p1"},
        paper_metadata={},
        documents={},
        metadata_fields=["title"],
        config={"embedding_batch_size": 1234},
        has_gpu=False,
        vector_store=MagicMock(),
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_all_papers_loaded_returns_early(mock_batch, request):
    """Test early return when all papers are already loaded."""
    article_data = request.getfixturevalue("articles")
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.loaded_papers = set(article_data.keys())

    load_all_papers(
        vector_store=vector_store,
        articles=article_data,
        call_id="test_call",
        config={"embedding_batch_size": 1000},
        has_gpu=False,
    )

    mock_batch.assert_not_called()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_skips_papers_without_pdf(mock_batch, request):
    """Test that papers without PDF URLs are skipped."""
    article_data = request.getfixturevalue("articles")
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.loaded_papers = {"p2"}  # p1 not loaded, p3 has no pdf

    load_all_papers(
        vector_store=vector_store,
        articles=article_data,
        call_id="test_call",
        config={"embedding_batch_size": 1000},
        has_gpu=False,
    )

    assert mock_batch.call_count == 1
    call_args = mock_batch.call_args[1]["papers_to_add"]
    assert len(call_args) == 1
    assert call_args[0][0] == "p1"


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_gpu_parameters_used(mock_batch, request):
    """Test GPU-based parameters are used if has_gpu is True."""
    article_data = request.getfixturevalue("articles")
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.loaded_papers = set()
    vector_store.has_gpu = True

    load_all_papers(
        vector_store=vector_store,
        articles=article_data,
        call_id="gpu_call",
        config={"embedding_batch_size": 2048},
        has_gpu=True,
    )

    args = mock_batch.call_args[1]
    assert args["has_gpu"] is True
    assert args["batch_size"] == 2048
    assert args["max_workers"] >= 4


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_cpu_parameters_used(mock_batch, request):
    """Test CPU-based parameters are used if has_gpu is False."""
    article_data = request.getfixturevalue("articles")
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.loaded_papers = set()
    vector_store.has_gpu = False

    load_all_papers(
        vector_store=vector_store,
        articles=article_data,
        call_id="cpu_call",
        config={"embedding_batch_size": 512},
        has_gpu=False,
    )

    args = mock_batch.call_args[1]
    assert args["has_gpu"] is False
    assert args["batch_size"] == 512
    assert args["max_workers"] >= 3
