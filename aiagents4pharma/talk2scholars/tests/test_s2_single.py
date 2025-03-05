import pytest
import requests
from types import SimpleNamespace

# Import the tool function from your module
from aiagents4pharma.talk2scholars.tools.s2.single_paper_rec import (
    get_single_paper_recommendations,
)
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import hydra

# --- Dummy Hydra Config Setup ---


class DummyHydraContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, traceback):
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
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP Error")
        return None


def test_dummy_response_no_error():
    # Create a DummyResponse with a successful status code.
    response = DummyResponse({"data": "success"}, status_code=200)
    # Calling raise_for_status should not raise an exception and should return None.
    assert response.raise_for_status() is None


def test_dummy_response_raise_error():
    # Create a DummyResponse with a failing status code.
    response = DummyResponse({"error": "fail"}, status_code=400)
    # Calling raise_for_status should raise an HTTPError.
    with pytest.raises(requests.HTTPError):
        response.raise_for_status()


def dummy_requests_get_success(url, params, timeout):
    # Record call parameters for assertions
    dummy_requests_get_success.called_url = url
    dummy_requests_get_success.called_params = params

    # Simulate a valid API response with three recommended papers;
    # one paper missing authors should be filtered out.
    dummy_data = {
        "recommendedPapers": [
            {
                "paperId": "paper1",
                "title": "Recommended Paper 1",
                "authors": ["Author A"],
                "year": 2020,
                "citationCount": 15,
                "url": "http://paper1",
                "externalIds": {"ArXiv": "arxiv1"},
            },
            {
                "paperId": "paper2",
                "title": "Recommended Paper 2",
                "authors": ["Author B"],
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
    # Simulate a response with an unexpected format (missing "recommendedPapers" key)
    return DummyResponse({"error": "Invalid format"})


def dummy_requests_get_no_recs(url, params, timeout):
    # Simulate a response with an empty recommendations list.
    return DummyResponse({"recommendedPapers": []})


def dummy_requests_get_exception(url, params, timeout):
    # Simulate a network/connection exception.
    raise requests.exceptions.RequestException("Connection error")


# --- Pytest Fixture to Patch Hydra ---
@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    # Patch hydra.initialize to return our dummy context manager.
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
        match="Unexpected response from Semantic Scholar API. Please retry the same query.",
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
        match="No recommendations returned from Semantic Scholar API. Please retry the same query.",
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
        match="Failed to connect to Semantic Scholar API. Please retry the same query.",
    ):
        get_single_paper_recommendations.run(input_data)
