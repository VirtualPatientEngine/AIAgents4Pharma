"""Tests for the PDF batch processor module."""

from unittest.mock import MagicMock, patch

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor import (
    add_papers_batch,
)


@pytest.fixture(name="args_fixture")
def _args_fixture():
    """Provides common arguments for tests."""
    return {
        "vector_store": MagicMock(),
        "loaded_papers": set(),
        "paper_metadata": {},
        "documents": {},
        "config": {"param": "value"},
        "metadata_fields": ["Title", "Author"],
        "has_gpu": False,
    }


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf")
def test_no_papers_to_add(mock_loader, args_fixture):
    """Test case where no papers are provided to add."""
    add_papers_batch(papers_to_add=[], **args_fixture)
    mock_loader.assert_not_called()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf")
def test_all_papers_already_loaded(mock_loader, args_fixture):
    """Test case where all papers are already loaded."""
    args_fixture["loaded_papers"].update(["p1", "p2"])
    add_papers_batch(
        papers_to_add=[("p1", "url1", {}), ("p2", "url2", {})],
        **args_fixture,
    )
    mock_loader.assert_not_called()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf")
def test_successful_batch_embedding(mock_loader, args_fixture):
    """Test case where papers are successfully loaded and embedded."""
    mock_loader.return_value = [
        MagicMock(page_content="Page 1"),
        MagicMock(page_content="Page 2"),
    ]

    mock_collection = MagicMock()
    mock_collection.num_entities = 2
    mock_collection.query.return_value = [{"paper_id": "p1"}]
    args_fixture["vector_store"].col = mock_collection

    add_papers_batch(
        papers_to_add=[("p1", "url1", {"Title": "Paper One"})],
        **args_fixture,
    )

    assert "p1" in args_fixture["paper_metadata"]
    assert "p1" in args_fixture["loaded_papers"]
    args_fixture["vector_store"].add_documents.assert_called_once()
    mock_collection.flush.assert_called()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf")
def test_empty_chunks_after_loading(mock_loader, args_fixture):
    """Test case where no chunks are returned after loading PDF."""
    mock_loader.return_value = []

    add_papers_batch(papers_to_add=[("p1", "url1", {})], **args_fixture)

    args_fixture["vector_store"].add_documents.assert_not_called()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf")
def test_vector_store_insert_failure(mock_loader, args_fixture):
    """Test case where vector store insertion fails."""
    mock_loader.return_value = [MagicMock(page_content="page")]

    def raise_error(*_, **__):
        raise RuntimeError("Vector store failed")

    args_fixture["vector_store"].add_documents.side_effect = raise_error

    mock_collection = MagicMock()
    args_fixture["vector_store"].col = mock_collection

    with pytest.raises(RuntimeError, match="Vector store failed"):
        add_papers_batch(papers_to_add=[("p1", "url1", {})], **args_fixture)
