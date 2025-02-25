import pytest
from unittest.mock import patch, MagicMock
import requests

from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Import the tool and the hydra config used by the tool.
# Adjust the relative import based on your project structure.
from ..tools.arxiv.download_pdf_arxiv import fetch_arxiv_paper, cfg


# Override the hydra config for the tests
@pytest.fixture(autouse=True)
def override_cfg():
    cfg.api_url = "http://arxiv.org/api/query"
    cfg.request_timeout = 10


@patch("requests.get")
def test_fetch_arxiv_paper_success(mock_get):
    """
    Test the successful download of a PDF.
    Simulate a metadata response with a valid PDF link and a PDF download response.
    """
    paper_id = "1905.02244"
    tool_call_id = "test123"
    # The state should contain an entry mapping the paper_id to its arXiv ID.
    state = {paper_id: {"arXiv ID": paper_id}}

    # Fake metadata XML response containing a pdf link.
    metadata_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <link title="pdf" href="http://arxiv.org/pdf/1905.02244.pdf"/>
      </entry>
    </feed>"""
    metadata_response = MagicMock()
    metadata_response.text = metadata_xml
    metadata_response.status_code = 200
    metadata_response.raise_for_status = MagicMock()

    # Fake PDF download response with binary content.
    pdf_data = b"fake pdf content"
    pdf_response = MagicMock()
    pdf_response.iter_content.return_value = [pdf_data]
    pdf_response.status_code = 200
    pdf_response.raise_for_status = MagicMock()

    # The first requests.get call returns the metadata response,
    # and the second returns the PDF response.
    mock_get.side_effect = [metadata_response, pdf_response]

    result = fetch_arxiv_paper(paper_id, tool_call_id, state)

    # Verify that a Command object is returned and state is updated correctly.
    assert isinstance(result, Command)
    update = result.update
    assert paper_id in update, "State update should include the paper_id"
    paper_update = update[paper_id]
    assert "pdf" in paper_update, "PDF data should be stored in state"
    assert paper_update["pdf"] == pdf_data, "PDF content does not match"
    assert "pdf_url" in paper_update, "PDF URL should be stored in state"
    assert paper_update["pdf_url"] == "http://arxiv.org/pdf/1905.02244.pdf"
    # Check that a success message was returned.
    messages = update.get("messages", [])
    assert any("Successfully downloaded PDF" in msg.content for msg in messages)


@patch("requests.get")
def test_fetch_arxiv_paper_failure(mock_get):
    """
    Test the failure scenario where the metadata response does not contain a PDF link.
    The tool should return a Command with a failure message.
    """
    paper_id = "1905.02244"
    tool_call_id = "test123"
    state = {paper_id: {"arXiv ID": paper_id}}

    # Fake metadata XML response with no PDF link.
    metadata_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry></entry>
    </feed>"""
    metadata_response = MagicMock()
    metadata_response.text = metadata_xml
    metadata_response.status_code = 200
    metadata_response.raise_for_status = MagicMock()

    # Only one requests.get call is made (for metadata).
    mock_get.return_value = metadata_response

    result = fetch_arxiv_paper(paper_id, tool_call_id, state)

    # Verify that a failure message is returned.
    assert isinstance(result, Command)
    update = result.update
    messages = update.get("messages", [])
    expected_message = f"Failed to download PDF for arXiv ID {paper_id}."
    assert any(expected_message in msg.content for msg in messages)
    # In this failure case, there should be no PDF data saved under paper_id.
    paper_info = update.get(paper_id, {})
    assert "pdf" not in paper_info


@patch("requests.get")
def test_fetch_arxiv_paper_api_failure(mock_get):
    """
    Test that an HTTP error in the metadata request raises an exception.
    """
    paper_id = "1905.02244"
    tool_call_id = "test123"
    state = {paper_id: {"arXiv ID": paper_id}}

    # Simulate an API failure with a non-200 status code.
    metadata_response = MagicMock()
    metadata_response.status_code = 500
    metadata_response.raise_for_status.side_effect = requests.HTTPError("Internal Server Error")

    mock_get.return_value = metadata_response

    with pytest.raises(requests.HTTPError, match="Internal Server Error"):
        fetch_arxiv_paper(paper_id, tool_call_id, state)
