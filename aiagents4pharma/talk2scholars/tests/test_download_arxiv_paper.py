"""
This module contains unit tests for the download_arxiv_paper tool.
"""
from unittest.mock import patch, MagicMock
import pytest
import requests

from langgraph.types import Command

from ..tools.paper_download.download_arxiv_paper import download_arxiv_paper


class DummyState(dict):
    """
    Dummy state class for simulating tool state with required callback attributes.
    """
    def __init__(self, *args, parent_run_id="dummy", **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_run_id = parent_run_id
        # Provide empty defaults for expected callback attributes:
        self.handlers = []
        self.inheritable_handlers = []
        self.tags = []
        self.inheritable_tags = []
        self.metadata = {}
        self.inheritable_metadata = {}


@patch("requests.get")
def test_download_arxiv_paper_success(mock_get):
    """
    Test the successful download of a PDF.
    Simulate a metadata response with a valid PDF link and a PDF download response.
    """
    arxiv_id = "1905.02244"
    tool_call_id = "test123"
    state = DummyState({arxiv_id: {"paper_id": arxiv_id}})

    # Fake metadata XML response containing a PDF link.
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

    # Use a flat dictionary for the tool input.
    input_dict = {
        "arxiv_id": arxiv_id,
        "tool_call_id": tool_call_id,
    }
    result = download_arxiv_paper.invoke(input_dict, state=state)

    # Verify that a Command object is returned and state is updated correctly.
    assert isinstance(result, Command)
    update = result.update

    # Verify that the pdf_data key is present.
    assert "pdf_data" in update, "State update should include the 'pdf_data' key"

    pdf_update = update["pdf_data"]
    # Check that the PDF data dictionary has the required keys.
    assert "arxiv_id" in pdf_update, "pdf_data should include the arxiv_id key"
    assert pdf_update["arxiv_id"] == arxiv_id, "arxiv_id does not match"
    assert "pdf_object" in pdf_update, "PDF data should be stored in pdf_data as pdf_object"
    assert pdf_update["pdf_object"] == pdf_data, "PDF content does not match"
    assert "pdf_url" in pdf_update, "PDF URL should be stored in pdf_data"
    assert pdf_update["pdf_url"] == "http://arxiv.org/pdf/1905.02244.pdf", "PDF URL does not match"

    # Verify that the messages key is present and contains a success message.
    messages = update.get("messages", [])
    assert any("Successfully downloaded PDF" in msg.content for msg in messages)


@patch("requests.get")
def test_download_arxiv_paper_failure(mock_get):
    """
    Test the failure scenario where the metadata response does not contain a PDF link.
    The tool should raise an Exception with an appropriate error message.
    """
    arxiv_id = "1905.02244"
    tool_call_id = "test123"
    state = DummyState({arxiv_id: {"arXiv ID": arxiv_id}})

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

    # Use a flat dictionary for the tool input.
    input_dict = {
        "arxiv_id": arxiv_id,
        "tool_call_id": tool_call_id,
    }

    with pytest.raises(RuntimeError) as exc_info:
        download_arxiv_paper.invoke(input_dict, state=state)

    expected_error = f"Failed to download PDF for arXiv ID {arxiv_id}."
    assert expected_error in str(exc_info.value)


@patch("requests.get")
def test_download_arxiv_paper_api_failure(mock_get):
    """
    Test that an HTTP error in the metadata request raises an exception.
    """
    arxiv_id = "1905.02244"
    tool_call_id = "test123"
    state = DummyState({arxiv_id: {"arXiv ID": arxiv_id}})

    # Simulate an API failure with a non-200 status code.
    metadata_response = MagicMock()
    metadata_response.status_code = 500
    metadata_response.raise_for_status.side_effect = requests.HTTPError("Internal Server Error")

    mock_get.return_value = metadata_response

    # Use a flat dictionary for the tool input.
    input_dict = {
        "arxiv_id": arxiv_id,
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(requests.HTTPError, match="Internal Server Error"):
        download_arxiv_paper.invoke(input_dict, state=state)
