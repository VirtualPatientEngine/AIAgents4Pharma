"""Unit tests for PDF document processing utilities."""

from unittest.mock import MagicMock, patch

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils.document_processor import (
    load_and_split_pdf,
)


@pytest.fixture(name="base_args_params")
def _base_args_params():
    """base_args_params fixture to provide common arguments for tests."""
    return {
        "paper_id": "P123",
        "pdf_url": "mock/path/to/paper.pdf",
        "paper_metadata": {"Title": "Test Paper", "Author": "A. Researcher"},
        "config": type("Config", (), {"chunk_size": 1000, "chunk_overlap": 200})(),
        "metadata_fields": ["Author"],
        "documents_dict": {},
    }


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.document_processor.PyPDFLoader")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.document_processor."
    "RecursiveCharacterTextSplitter"
)
def test_load_and_split_pdf_success(mock_splitter_cls, mock_loader_cls, base_args_params):
    """load_and_split_pdf should load and split PDF correctly."""
    mock_doc = MagicMock()
    mock_doc.metadata = {"page": 1}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc]
    mock_loader_cls.return_value = mock_loader

    mock_splitter = MagicMock()
    chunk1 = MagicMock()
    chunk1.metadata = {"page": 1}
    mock_splitter.split_documents.return_value = [chunk1]
    mock_splitter_cls.return_value = mock_splitter

    chunks = load_and_split_pdf(**base_args_params)

    assert len(chunks) == 1
    assert "P123_0" in base_args_params["documents_dict"]
    stored_chunk = base_args_params["documents_dict"]["P123_0"]
    assert stored_chunk.metadata["paper_id"] == "P123"
    assert stored_chunk.metadata["title"] == "Test Paper"
    assert stored_chunk.metadata["chunk_id"] == 0
    assert stored_chunk.metadata["page"] == 1
    assert stored_chunk.metadata["source"] == base_args_params["pdf_url"]
    assert stored_chunk.metadata["Author"] == "A. Researcher"


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.document_processor.PyPDFLoader")
def test_load_and_split_pdf_raises_if_config_missing(mock_loader_cls, base_args_params):
    """load_and_split_pdf should raise ValueError if config is None."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = [MagicMock()]
    mock_loader_cls.return_value = mock_loader

    base_args_params["config"] = None
    with pytest.raises(
        ValueError, match="Configuration is required for text splitting in Vectorstore."
    ):
        load_and_split_pdf(**base_args_params)
