"""
Unit tests for arXiv paper downloading functionality, including:
- AbstractPaperDownloader (base class)
- ArxivPaperDownloader (arXiv-specific implementation)
- download_arxiv_paper tool function.
"""

from unittest.mock import patch, MagicMock
import pytest
import requests
from requests.exceptions import HTTPError
from langgraph.types import Command
from langchain_core.messages import ToolMessage

# Import the classes and function under test
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
        class_obj()


@pytest.fixture(name="arxiv_downloader_fixture")
@pytest.mark.usefixtures("mock_hydra_config_setup")
def fixture_arxiv_downloader():
    """
    Provides an ArxivPaperDownloader instance with a mocked Hydra config.
    """
    return ArxivPaperDownloader()


def test_fetch_metadata_success(arxiv_downloader_fixture,):
    """
    Ensures fetch_metadata retrieves XML data correctly, given a successful HTTP response.
    """
    mock_response = MagicMock()
    mock_response.text = "<xml>Mock ArXiv Metadata</xml>"
    mock_response.raise_for_status = MagicMock()

    with patch.object(requests, "get", return_value=mock_response) as mock_get:
        paper_id = "1234.5678"
        result = arxiv_downloader_fixture.fetch_metadata(paper_id)
        mock_get.assert_called_once_with(
            "http://export.arxiv.org/api/query?search_query=id:1234.5678&start=0&max_results=1",
            timeout=10,
        )
        assert result["xml"] == "<xml>Mock ArXiv Metadata</xml>"


def test_fetch_metadata_http_error(arxiv_downloader_fixture):
    """
    Validates that fetch_metadata raises HTTPError when the response indicates a failure.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = HTTPError("Mocked HTTP failure")

    with patch.object(requests, "get", return_value=mock_response):
        with pytest.raises(HTTPError):
            arxiv_downloader_fixture.fetch_metadata("invalid_id")


def test_download_pdf_success(arxiv_downloader_fixture):
    """
    Tests that download_pdf fetches the PDF link from metadata and successfully
    retrieves the binary content.
    """
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

    with patch.object(arxiv_downloader_fixture, "fetch_metadata", return_value=mock_metadata):
        with patch.object(requests, "get", return_value=mock_pdf_response) as mock_get:
            result = arxiv_downloader_fixture.download_pdf("1234.5678")
            assert result["pdf_object"] == b"FAKE_PDF_CONTENT"
            assert result["pdf_url"] == "http://test.arxiv.org/pdf/1234.5678v1.pdf"
            assert result["arxiv_id"] == "1234.5678"
            mock_get.assert_called_once_with(
                "http://test.arxiv.org/pdf/1234.5678v1.pdf",
                stream=True,
                timeout=10,
            )


def test_download_pdf_no_pdf_link(arxiv_downloader_fixture):
    """
    Ensures a RuntimeError is raised if no <link> with title="pdf" is found in the XML.
    """
    mock_metadata = {"xml": "<feed></feed>"}

    with patch.object(arxiv_downloader_fixture, "fetch_metadata", return_value=mock_metadata):
        with pytest.raises(RuntimeError, match="Failed to download PDF"):
            arxiv_downloader_fixture.download_pdf("1234.5678")


def test_download_arxiv_paper_tool_success(arxiv_downloader_fixture):
    """
    Validates download_arxiv_paper orchestrates the ArxivPaperDownloader correctly,
    returning a Command with PDF data and success messages.
    """
    mock_metadata = {"xml": "<mockxml></mockxml>"}
    mock_pdf_response = {
        "pdf_object": b"FAKE_PDF_CONTENT",
        "pdf_url": "http://test.arxiv.org/mock.pdf",
        "arxiv_id": "9999.8888",
    }

    with patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input."
        "ArxivPaperDownloader",
        return_value=arxiv_downloader_fixture,
    ):
        with patch.object(arxiv_downloader_fixture, "fetch_metadata", return_value=mock_metadata):
            with patch.object(
                arxiv_downloader_fixture,
                "download_pdf",
                return_value=mock_pdf_response,
            ):
                command_result = download_arxiv_paper.invoke(
                    {"arxiv_id": "9999.8888", "tool_call_id": "test_tool_call"}
                )

                assert isinstance(command_result, Command)
                assert "pdf_data" in command_result.update
                assert command_result.update["pdf_data"] == mock_pdf_response

                messages = command_result.update.get("messages", [])
                assert len(messages) == 1
                assert isinstance(messages[0], ToolMessage)
                assert "Successfully downloaded PDF" in messages[0].content
                assert "9999.8888" in messages[0].content
