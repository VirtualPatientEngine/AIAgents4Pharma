# File: tests/test_arxiv_tools.py

import pytest
import requests
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from aiagents4pharma.talk2scholars.tools.paper_download.abstract_downloader import (
    AbstractPaperDownloader,
)
from aiagents4pharma.talk2scholars.tools.paper_download.arxiv_downloader import (
    ArxivPaperDownloader,
)
from aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input import (
    download_arxiv_paper,
)

@pytest.mark.parametrize("class_obj", [AbstractPaperDownloader])
def test_abstract_downloader_cannot_be_instantiated(class_obj):
    """
    Validates that AbstractPaperDownloader is indeed abstract and raises TypeError
    if anyone attempts to instantiate it directly.
    """
    with pytest.raises(TypeError):
        _ = class_obj()


@pytest.fixture
def mock_hydra_config(mocker):
    """
    Mocks out Hydra's initialize() and compose() calls to prevent real configuration loading
    during tests. This keeps tests reliable and independent of external config.
    """
    # Patch the Hydra initialization process:
    mocker.patch("hydra.initialize")
    # Patch compose() to return a mock config with expected structure:
    mocker.patch(
        "hydra.compose",
        return_value=mocker.MagicMock(
            tools=mocker.MagicMock(
                download_arxiv_paper=mocker.MagicMock(
                    api_url="http://test.arxiv.org/mockapi",
                    request_timeout=10,
                )
            )
        ),
    )


@pytest.fixture
def arxiv_downloader(mock_hydra_config):
    """
    Provides an ArxivPaperDownloader instance with a mocked Hydra config.
    """
    return ArxivPaperDownloader()


def test_fetch_metadata_success(arxiv_downloader):
    """
    Ensures fetch_metadata retrieves XML data correctly, given a successful HTTP response.
    """
    mock_response = MagicMock()
    mock_response.text = "<xml>Mock ArXiv Metadata</xml>"
    mock_response.raise_for_status = MagicMock()

    with patch.object(requests, "get", return_value=mock_response) as mock_get:
        paper_id = "1234.5678"
        result = arxiv_downloader.fetch_metadata(paper_id)
        mock_get.assert_called_once_with(
            f"http://test.arxiv.org/mockapi?search_query=id:{paper_id}&start=0&max_results=1",
            timeout=10,
        )
        assert result["xml"] == "<xml>Mock ArXiv Metadata</xml>"


def test_fetch_metadata_http_error(arxiv_downloader):
    """
    Validates that fetch_metadata raises HTTPError when the response indicates a failure.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = HTTPError("Mocked HTTP failure")

    with patch.object(requests, "get", return_value=mock_response):
        with pytest.raises(HTTPError):
            arxiv_downloader.fetch_metadata("invalid_id")


def test_download_pdf_success(arxiv_downloader):
    """
    Tests that download_pdf successfully fetches the PDF link from metadata and retrieves the binary.
    """
    # Mock metadata to include a valid PDF link:
    mock_metadata = {
        "xml": """
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <link title="pdf" href="http://test.arxiv.org/pdf/1234.5678v1.pdf"/>
            </entry>
        </feed>
        """
    }

    mock_pdf_response = MagicMock()
    mock_pdf_response.raise_for_status = MagicMock()
    mock_pdf_response.iter_content = lambda chunk_size: [b"FAKE_PDF_CONTENT"]

    with patch.object(arxiv_downloader, "fetch_metadata", return_value=mock_metadata):
        with patch.object(requests, "get", return_value=mock_pdf_response) as mock_get:
            result = arxiv_downloader.download_pdf("1234.5678")
            assert result["pdf_object"] == b"FAKE_PDF_CONTENT"
            assert result["pdf_url"] == "http://test.arxiv.org/pdf/1234.5678v1.pdf"
            assert result["arxiv_id"] == "1234.5678"
            mock_get.assert_called_once_with(
                "http://test.arxiv.org/pdf/1234.5678v1.pdf",
                stream=True,
                timeout=10,
            )


def test_download_pdf_no_pdf_link(arxiv_downloader):
    """
    Ensures a RuntimeError is raised if no <link> with title="pdf" is found in the XML.
    """
    mock_metadata = {"xml": "<feed></feed>"}

    with patch.object(arxiv_downloader, "fetch_metadata", return_value=mock_metadata):
        with pytest.raises(RuntimeError, match="Failed to download PDF"):
            arxiv_downloader.download_pdf("1234.5678")


def test_download_arxiv_paper_tool_success(arxiv_downloader):
    """
    Validates the download_arxiv_paper function orchestrates the ArxivPaperDownloader correctly,
    returning a Command with PDF data and success messages.
    """

    mock_metadata = {"xml": "<mockxml></mockxml>"}
    mock_pdf_response = {
        "pdf_object": b"FAKE_PDF_CONTENT",
        "pdf_url": "http://test.arxiv.org/mock.pdf",
        "arxiv_id": "9999.8888",
    }

    # Patch the ArxivPaperDownloader constructor to return our fixture
    with patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.ArxivPaperDownloader",
        return_value=arxiv_downloader,
    ):
        with patch.object(arxiv_downloader, "fetch_metadata", return_value=mock_metadata):
            with patch.object(arxiv_downloader, "download_pdf", return_value=mock_pdf_response):
                # Invoke the tool by passing a SINGLE dictionary argument:
                command_result = download_arxiv_paper.invoke({
                    "arxiv_id": "9999.8888",
                    "tool_call_id": "test_tool_call",
                })

                # Validate structure
                assert isinstance(command_result, Command)
                assert "pdf_data" in command_result.update
                assert command_result.update["pdf_data"] == mock_pdf_response

                messages = command_result.update.get("messages", [])
                assert len(messages) == 1
                assert isinstance(messages[0], ToolMessage)
                assert "Successfully downloaded PDF" in messages[0].content
                assert "9999.8888" in messages[0].content
