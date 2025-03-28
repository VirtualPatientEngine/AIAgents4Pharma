"""
Unit tests for S2 tools functionality.
"""

from types import SimpleNamespace
import pytest
import requests
import hydra
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from aiagents4pharma.talk2scholars.tools.s2.single_paper_rec import (
    get_single_paper_recommendations,
)
from aiagents4pharma.talk2scholars.tools.s2.utils import single_helper

# --- Dummy Hydra Config Setup ---


class DummyHydraContext:
    """
    A dummy context manager for mocking Hydra's initialize and compose functions.
    """

    def __enter__(self):
        return None

    def __exit__(self, exc_val, exc_type, traceback):
        pass


# Create a dummy configuration that mimics the expected Hydra config
dummy_config = SimpleNamespace(
    tools=SimpleNamespace(
        single_paper_recommendation=SimpleNamespace(
            api_endpoint="http://dummy.endpoint",
            api_fields=["paperId", "title", "authors"],
            recommendation_params=SimpleNamespace(from_pool="default_pool"),
            request_timeout=10,
        )
    )
)

# --- Dummy Response Classes and Functions for requests.get ---


class DummyResponse:
    """
    A dummy response class for mocking HTTP responses.
    """

    def __init__(self, json_data, status_code=200):
        """Initialize a DummyResponse with the given JSON data and status code."""
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        """Return the JSON data from the response."""
        return self._json_data

    def raise_for_status(self):
        """Raise an HTTP error for status codes >= 400."""
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP Error")


def test_dummy_response_no_error():
    """Test DummyResponse does not raise an error for 200 status code."""
    response = DummyResponse({"data": "success"}, status_code=200)
    # Calling raise_for_status should not raise an exception and should return None.
    assert response.raise_for_status() is None


def test_dummy_response_raise_error():
    """Test DummyResponse raises an HTTPError for status codes >= 400."""
    response = DummyResponse({"error": "fail"}, status_code=400)
    # Calling raise_for_status should raise an HTTPError.
    with pytest.raises(requests.HTTPError):
        response.raise_for_status()


def dummy_requests_get_success(url, params, timeout):
    """
    Dummy function to simulate a successful API request returning recommended papers.
    """
    dummy_requests_get_success.called_url = url
    dummy_requests_get_success.called_params = params
    dummy_requests_get_success.called_timeout = timeout

    # Simulate a valid API response with three recommended papers;
    # one paper missing authors should be filtered out.
    dummy_data = {
        "recommendedPapers": [
            {
                "paperId": "paper1",
                "title": "Recommended Paper 1",
                "authors": [{"name": "Author A", "authorId": "A1"}],
                "year": 2020,
                "citationCount": 15,
                "url": "http://paper1",
                "externalIds": {"ArXiv": "arxiv1"},
            },
            {
                "paperId": "paper2",
                "title": "Recommended Paper 2",
                "authors": [{"name": "Author B", "authorId": "B1"}],
                "year": 2021,
                "citationCount": 25,
                "url": "http://paper2",
                "externalIds": {},
            },
            {
                "paperId": "paper3",
                "title": "Recommended Paper 3",
                "authors": None,  # This paper should be filtered out.
                "year": 2022,
                "citationCount": 35,
                "url": "http://paper3",
                "externalIds": {"ArXiv": "arxiv3"},
            },
        ]
    }
    return DummyResponse(dummy_data)


def dummy_requests_get_unexpected(url, params, timeout):
    """
    Dummy function to simulate an API response with an unexpected format.
    """
    dummy_requests_get_unexpected.called_url = url
    dummy_requests_get_unexpected.called_params = params
    dummy_requests_get_unexpected.called_timeout = timeout
    return DummyResponse({"error": "Invalid format"})


def dummy_requests_get_no_recs(url, params, timeout):
    """
    Dummy function to simulate an API response returning no recommendations.
    """
    dummy_requests_get_no_recs.called_url = url
    dummy_requests_get_no_recs.called_params = params
    dummy_requests_get_no_recs.called_timeout = timeout
    return DummyResponse({"recommendedPapers": []})


def dummy_requests_get_exception(url, params, timeout):
    """
    Dummy function to simulate a request exception (e.g., network failure).
    """
    dummy_requests_get_exception.called_url = url
    dummy_requests_get_exception.called_params = params
    dummy_requests_get_exception.called_timeout = timeout
    raise requests.exceptions.RequestException("Connection error")


# --- Pytest Fixture to Patch Hydra ---
@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    """Patch Hydra's initialize and compose functions with dummy implementations."""
    monkeypatch.setattr(
        hydra, "initialize", lambda version_base, config_path: DummyHydraContext()
    )
    # Patch hydra.compose to return our dummy config.
    monkeypatch.setattr(hydra, "compose", lambda config_name, overrides: dummy_config)


# --- Test Cases ---


def test_single_paper_rec_success(monkeypatch):
    """
    Test that get_single_paper_recommendations returns a valid Command object
    when the API response is successful. Also, ensure that recommendations missing
    required fields (like authors) are filtered out.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_success)

    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_id": "12345",
        "tool_call_id": tool_call_id,
        "limit": 3,
        "year": "2020",
    }
    # Invoke the tool using .run() with a single dictionary as input.
    result = get_single_paper_recommendations.run(input_data)

    # Validate that the result is a Command with the expected structure.
    assert isinstance(result, Command)
    update = result.update
    assert "papers" in update

    papers = update["papers"]
    # Papers with valid 'title' and 'authors' should be included.
    assert "paper1" in papers
    assert "paper2" in papers
    # Paper "paper3" is missing authors and should be filtered out.
    assert "paper3" not in papers

    # Check that a ToolMessage is included in the messages.
    messages = update.get("messages", [])
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ToolMessage)
    assert "Recommendations based on the single paper were successful" in msg.content

    # Verify that the correct parameters were sent to requests.get.
    called_params = dummy_requests_get_success.called_params
    assert called_params["limit"] == 3  # limited to min(limit, 500)
    # "fields" should be a comma-separated string from the dummy config.
    assert called_params["fields"] == "paperId,title,authors"
    # Check that the "from" parameter is set from our dummy config.
    assert called_params["from"] == "default_pool"
    # The year parameter should be present.
    assert called_params["year"] == "2020"


def test_single_paper_rec_unexpected_format(monkeypatch):
    """
    Test that get_single_paper_recommendations raises a RuntimeError when the API
    response does not include the expected 'recommendedPapers' key.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_unexpected)
    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_id": "12345",
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
        get_single_paper_recommendations.run(input_data)


def test_single_paper_rec_no_recommendations(monkeypatch):
    """
    Test that get_single_paper_recommendations raises a RuntimeError when the API
    returns no recommendations.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_no_recs)
    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_id": "12345",
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(
        RuntimeError,
        match=(
            "No recommendations were found for your query. Consider refining your search "
            "by using more specific keywords or different terms."
        ),
    ):
        get_single_paper_recommendations.run(input_data)


def test_single_paper_rec_requests_exception(monkeypatch):
    """
    Test that get_single_paper_recommendations raises a RuntimeError when requests.get
    throws an exception.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_exception)
    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_id": "12345",
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(
        RuntimeError,
        match="Failed to connect to Semantic Scholar API after 10 attempts."
        "Please retry the same query.",
    ):
        get_single_paper_recommendations.run(input_data)


def test_single_paper_rec_no_response(monkeypatch):
    """
    Test that get_single_paper_recommendations raises a RuntimeError
    when no response is obtained from the API.
    This is simulated by patching 'range' in the module namespace
    of single_helper to return an empty iterator, so that the for-loop
    in _fetch_recommendations never iterates and response remains None.
    """
    # Patch 'range' in the module globals of single_helper.
    monkeypatch.setitem(single_helper.__dict__, "range", lambda x: iter([]))

    tool_call_id = "test_tool_call_id"
    input_data = {
        "paper_id": "12345",
        "tool_call_id": tool_call_id,
    }
    with pytest.raises(
        RuntimeError, match="Failed to obtain a response from the Semantic Scholar API."
    ):
        get_single_paper_recommendations.run(input_data)
