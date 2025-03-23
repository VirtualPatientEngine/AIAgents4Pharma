"""Unit tests for PubMed paper downloader and tool wrapper."""

from unittest.mock import patch, MagicMock
import pytest
from requests.exceptions import HTTPError
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from ..tools.paper_download.pubmed_downloader import (
     PubMedPaperDownloader,
)
from ..tools.paper_download.download_pubmed_input import (
     download_pubmed_paper,
)


@pytest.fixture
def pubmed_downloader_fixture(): # pylint: disable=redefined-outer-name
    """Fixture for mocking Hydra-based PubMedPaperDownloader instantiation."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.paper_download."
            "pubmed_downloader.hydra.initialize"
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.paper_download."
            "pubmed_downloader.hydra.compose"
        ),
    ):
        return PubMedPaperDownloader()


def test_fetch_metadata_success(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test that metadata is fetched successfully for a valid PMID."""
    mock_response = MagicMock()
    mock_response.text = "<xml>Mock PubMed Metadata</xml>"
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response) as mock_get:
        pmid = "123456"
        result = pubmed_downloader_fixture.fetch_metadata(pmid)
        mock_get.assert_called_once()
        assert result["xml"] == "<xml>Mock PubMed Metadata</xml>"


def test_fetch_metadata_http_error(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test that HTTPError is raised for invalid metadata fetch."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = HTTPError("Mocked HTTP failure")

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(HTTPError):
            pubmed_downloader_fixture.fetch_metadata("invalid_id")


def test_get_pmcid_from_pmid_success(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test PMCID is correctly extracted from ELink API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "linksets": [{"linksetdbs": [{"links": ["1234567"]}]}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        pmcid = pubmed_downloader_fixture.get_pmcid_from_pmid("123456")
        assert pmcid == "1234567"


def test_get_pmcid_from_pmid_failure(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test that None is returned when PMCID is missing in ELink response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        pmcid = pubmed_downloader_fixture.get_pmcid_from_pmid("invalid")
        assert pmcid is None


def test_download_pdf_success(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test successful PDF download with a valid PMCID."""
    with patch.object(pubmed_downloader_fixture, "get_pmcid_from_pmid",
                      return_value="1234567"):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = lambda chunk_size: [b"PDF_DATA"]
        with patch("requests.get", return_value=mock_response):
            result = pubmed_downloader_fixture.download_pdf("123456")
            assert result["pdf_object"] == b"PDF_DATA"
            assert result["pmcid"] == "1234567"


def test_download_pdf_no_pmcid(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test RuntimeError is raised if PMCID cannot be resolved."""
    with patch.object(pubmed_downloader_fixture, "get_pmcid_from_pmid",
                      return_value=None):
        with pytest.raises(RuntimeError, match="Could not resolve PMCID"):
            pubmed_downloader_fixture.download_pdf("123456")


def test_download_pdf_not_open_access(pubmed_downloader_fixture): # pylint: disable=redefined-outer-name
    """Test RuntimeError if PDF is not available or access is denied."""
    with patch.object(pubmed_downloader_fixture, "get_pmcid_from_pmid",
                      return_value="1234567"):
        mock_response = MagicMock()
        mock_response.status_code = 403
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(RuntimeError,
                               match="No PDF found or access denied"):
                pubmed_downloader_fixture.download_pdf("123456")


def test_download_pubmed_paper_tool_success():
    """Test that tool wrapper returns Command and ToolMessage correctly."""
    mock_response = {
        "pdf_object": b"FAKE_PDF_CONTENT",
        "pdf_url": "http://mock.url/pdf",
        "pmid": "999999",
        "pmcid": "1234567",
    }

    with patch("aiagents4pharma.talk2scholars.tools.paper_download."
               "download_pubmed_input.PubMedPaperDownloader") as mock_downloader:
        instance = mock_downloader.return_value
        instance.download_pdf.return_value = mock_response

        result = download_pubmed_paper.invoke(
            {"pmid": "999999", "tool_call_id": "tool_call_test"}
        )

        assert isinstance(result, Command)
        assert result.update["pdf_data"] == mock_response

        messages = result.update.get("messages", [])
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert "Successfully downloaded PDF for PMID 999999" in messages[0].content
