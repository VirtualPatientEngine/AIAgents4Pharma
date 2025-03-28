"""
Unit tests for S2 tools functionality.
"""

import json
from types import SimpleNamespace
import pytest
import requests
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import hydra
from aiagents4pharma.talk2scholars.tools.s2.multi_paper_rec import (
    get_multi_paper_recommendations,
)
from aiagents4pharma.talk2scholars.tools.s2.utils import multi_helper

# --- Dummy Hydra Config Setup ---


class DummyHydraContext:
    """dummy context manager for mocking Hydra's initialize and compose functions."""

    def __enter__(self):
        """enter function that returns None."""
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        """exit function that does nothing."""
        return None


# Create a dummy configuration that mimics the expected hydra config.
dummy_config = SimpleNamespace(
    tools=SimpleNamespace(
        multi_paper_recommendation=SimpleNamespace(
            api_endpoint="http://dummy.endpoint/multi",
            headers={"Content-Type": "application/json"},
            api_fields=["paperId", "title", "authors"],
            request_timeout=10,
        )
    )
)

# --- Dummy Response Classes and Functions for requests.post ---


class DummyResponse:
    """A dummy response class for mocking HTTP responses."""

    def __init__(self, json_data, status_code=200):
        """Initialize a DummyResponse with the given JSON data and status code."""
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        """Return the JSON data from the response."""
        return self._json_data

    def raise_for_status(self):
        """raise an HTTP error for status codes >= 400."""
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP Error")


def test_dummy_response_no_error():
    """Test that raise_for_status does not raise an exception for a successful response."""
    # Create a DummyResponse with a successful status code.
    response = DummyResponse({"data": "success"}, status_code=200)
    # Calling raise_for_status should not raise an exception and should return None.
    assert response.raise_for_status() is None


def test_dummy_response_raise_error():
    """Test that raise_for_status raises an exception for a failing response."""
    # Create a DummyResponse with a failing status code.
    response = DummyResponse({"error": "fail"}, status_code=400)
    # Calling raise_for_status should raise an HTTPError.
    with pytest.raises(requests.HTTPError):
        response.raise_for_status()


def dummy_requests_post_success(url, headers, params, data, timeout):
    """dummy_requests_post_success"""
    # Record call parameters for assertions.
    dummy_requests_post_success.called_url = url
    dummy_requests_post_success.called_headers = headers
    dummy_requests_post_success.called_params = params
    dummy_requests_post_success.called_data = data
    dummy_requests_post_success.called_timeout = timeout

    # Simulate a valid API response with three recommended papers;
    # one paper missing authors should be filtered out.
    dummy_data = {
        "recommendedPapers": [
            {
                "paperId": "paperA",
                "title": "Multi Rec Paper A",
                "authors": [{"name": "Author X", "authorId": "AX"}],
                "year": 2019,
                "citationCount": 12,
                "url": "http://paperA",
                "externalIds": {"ArXiv": "arxivA"},
            },
            {
                "paperId": "paperB",
                "title": "Multi Rec Paper B",
                "authors": [{"name": "Author Y", "authorId": "AY"}],
                "year": 2020,
                "citationCount": 18,
                "url": "http://paperB",
                "externalIds": {},
            },
            {
                "paperId": "paperC",
                "title": "Multi Rec Paper C",
                "authors": None,  # This paper should be filtered out.
                "year": 2021,
                "citationCount": 25,
                "url": "http://paperC",
                "externalIds": {"ArXiv": "arxivC"},
            },
        ]
    }
    return DummyResponse(dummy_data)


def dummy_requests_post_unexpected(url, headers, params, data, timeout):
    """dummy_requests_post_unexpected"""
    dummy_requests_post_unexpected.called_url = url
    dummy_requests_post_unexpected.called_headers = headers
    dummy_requests_post_unexpected.called_params = params
    dummy_requests_post_unexpected.called_data = data
    dummy_requests_post_unexpected.called_timeout = timeout
    # Simulate a response missing the 'recommendedPapers' key.
    return DummyResponse({"error": "Invalid format"})


def dummy_requests_post_no_recs(url, headers, params, data, timeout):
    """dummy_requests_post_no_recs"""
    dummy_requests_post_no_recs.called_url = url
    dummy_requests_post_no_recs.called_headers = headers
    dummy_requests_post_no_recs.called_params = params
    dummy_requests_post_no_recs.called_data = data
    dummy_requests_post_no_recs.called_timeout = timeout
    # Simulate a response with an empty recommendations list.
    return DummyResponse({"recommendedPapers": []})


def dummy_requests_post_exception(url, headers, params, data, timeout):
    """dummy_requests_post_exception"""
    dummy_requests_post_exception.called_url = url
    dummy_requests_post_exception.called_headers = headers
    dummy_requests_post_exception.called_params = params
    dummy_requests_post_exception.called_data = data
    dummy_requests_post_exception.called_timeout = timeout
    # Simulate a network exception.
    raise requests.exceptions.RequestException("Connection error")


# --- Pytest Fixture to Patch Hydra ---
@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    """Patch Hydra's initialize and compose functions to return dummy objects."""
    # Patch hydra.initialize to return our dummy context manager.
    monkeypatch.setattr(
        hydra, "initialize", lambda version_base, config_path: DummyHydraContext()
    )
    # Patch hydra.compose to return our dummy config.
    monkeypatch.setattr(hydra, "compose", lambda config_name, overrides: dummy_config)


# --- Test Cases ---


def test_multi_paper_rec_success(monkeypatch):
    """
    Test that get_multi_paper_recommendations returns a valid Command object
    when the API response is successful. Also, ensure that recommendations missing
    required fields (like authors) are filtered out.
    """
    monkeypatch.setattr(requests, "post", dummy_requests_post_success)

    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_ids": ["p1", "p2"],
        "tool_call_id": tool_call_id,
        "limit": 2,
        "year": "2020",
    }
    # Call the tool using .run() with a dictionary input.
    result = get_multi_paper_recommendations.run(input_data)

    # Validate that the result is a Command with the expected update structure.
    assert isinstance(result, Command)
    update = result.update
    assert "multi_papers" in update

    papers = update["multi_papers"]
    # Papers with valid 'title' and 'authors' should be included.
    assert "paperA" in papers
    assert "paperB" in papers
    # Paper "paperC" is missing authors and should be filtered out.
    assert "paperC" not in papers

    # Check that a ToolMessage is included in the messages.
    messages = update.get("messages", [])
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ToolMessage)
    assert "Recommendations based on multiple papers were successful" in msg.content

    # Verify that the correct parameters were sent to requests.post.
    called_params = dummy_requests_post_success.called_params
    assert called_params["limit"] == 2  # Should be min(limit, 500)
    assert called_params["fields"] == "paperId,title,authors"
    # The year parameter should be present.
    assert called_params["year"] == "2020"

    # Also check the payload sent in the data.
    sent_payload = json.loads(dummy_requests_post_success.called_data)
    assert sent_payload["positivePaperIds"] == ["p1", "p2"]
    assert sent_payload["negativePaperIds"] == []


def test_multi_paper_rec_unexpected_format(monkeypatch):
    """
    Test that get_multi_paper_recommendations raises a RuntimeError when the API
    response does not include the expected 'recommendedPapers' key.
    """
    monkeypatch.setattr(requests, "post", dummy_requests_post_unexpected)
    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_ids": ["p1", "p2"],
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(
        RuntimeError,
        match=(
            "Unexpected response from Semantic Scholar API. The results could not be "
            "retrieved due to an unexpected format. "
            "Please modify your search query and try again."
        ),
    ):
        get_multi_paper_recommendations.run(input_data)


def test_multi_paper_rec_no_recommendations(monkeypatch):
    """
    Test that get_multi_paper_recommendations raises a RuntimeError when the API
    returns no recommendations.
    """
    monkeypatch.setattr(requests, "post", dummy_requests_post_no_recs)
    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_ids": ["p1", "p2"],
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(
        RuntimeError,
        match=(
            "No recommendations were found for your query. Consider refining your search "
            "by using more specific keywords or different terms."
        ),
    ):
        get_multi_paper_recommendations.run(input_data)


def test_multi_paper_rec_requests_exception(monkeypatch):
    """
    Test that get_multi_paper_recommendations raises a RuntimeError when requests.post
    throws an exception.
    """
    monkeypatch.setattr(requests, "post", dummy_requests_post_exception)
    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_ids": ["p1", "p2"],
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(
        RuntimeError,
        match="Failed to connect to Semantic Scholar API after 10 attempts."
        "Please retry the same query.",
    ):
        get_multi_paper_recommendations.run(input_data)


def test_multi_paper_rec_no_response(monkeypatch):
    """
    Test that get_multi_paper_recommendations raises a RuntimeError
    when no response is obtained. This is simulated by patching 'range'
    in the module namespace of multi_helper to return an empty iterator,
    so that the for loop in _fetch_recommendations never iterates and
    self.response remains None.
    """
    # Inject a patched 'range' into the multi_helper module's dictionary.
    monkeypatch.setitem(multi_helper.__dict__, "range", lambda x: iter([]))

    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_ids": ["p1", "p2"],
        "tool_call_id": tool_call_id,
    }

    with pytest.raises(
        RuntimeError, match="Failed to obtain a response from the Semantic Scholar API."
    ):
        get_multi_paper_recommendations.run(input_data)
