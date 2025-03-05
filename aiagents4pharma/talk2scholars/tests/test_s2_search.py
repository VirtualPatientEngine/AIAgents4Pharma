import pytest
import requests
from types import SimpleNamespace

# Import the search_tool from your module
from aiagents4pharma.talk2scholars.tools.s2.search import search_tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import hydra

# --- Dummy Hydra Config Setup ---


class DummyHydraContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        pass


# Create a dummy configuration that mimics the expected hydra config
dummy_config = SimpleNamespace(
    tools=SimpleNamespace(
        search=SimpleNamespace(
            api_endpoint="http://dummy.endpoint",
            api_fields=["paperId", "title", "authors"],
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

    # Simulate a valid API response with three papers;
    # one paper missing authors should be filtered out.
    dummy_data = {
        "data": [
            {
                "paperId": "1",
                "title": "Paper 1",
                "authors": ["Author A"],
                "year": 2020,
                "citationCount": 10,
                "url": "http://paper1",
                "externalIds": {"ArXiv": "arxiv1"},
            },
            {
                "paperId": "2",
                "title": "Paper 2",
                "authors": ["Author B"],
                "year": 2021,
                "citationCount": 20,
                "url": "http://paper2",
                "externalIds": {},
            },
            {
                "paperId": "3",
                "title": "Paper 3",
                "authors": None,  # This paper should be filtered out.
                "year": 2022,
                "citationCount": 30,
                "url": "http://paper3",
                "externalIds": {"ArXiv": "arxiv3"},
            },
        ]
    }
    return DummyResponse(dummy_data)


def dummy_requests_get_no_data(url, params, timeout):
    # Simulate a response with an unexpected format (missing "data" key)
    return DummyResponse({"error": "Invalid format"})


def dummy_requests_get_no_papers(url, params, timeout):
    # Simulate a response with an empty papers list.
    return DummyResponse({"data": []})


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


def test_search_tool_success(monkeypatch):
    """
    Test that search_tool returns a valid Command object when the API response is successful.
    Also checks that papers without required fields are filtered out.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_success)

    tool_call_id = "test_tool_call_id"
    # Invoke using .run() with a dictionary input.
    result = search_tool.run(
        {
            "query": "machine learning",
            "tool_call_id": tool_call_id,
            "limit": 3,
            "year": "2020",
        }
    )

    # Check that a Command is returned with the expected update structure.
    assert isinstance(result, Command)
    update = result.update
    assert "papers" in update

    papers = update["papers"]
    # Papers with valid 'title' and 'authors' should be included.
    assert "1" in papers
    assert "2" in papers
    # Paper "3" is missing authors and should be filtered out.
    assert "3" not in papers

    # Check that a ToolMessage is included in the messages.
    messages = update.get("messages", [])
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ToolMessage)
    assert "Number of papers found:" in msg.content

    # Verify that the correct parameters were sent to requests.get.
    called_params = dummy_requests_get_success.called_params
    assert called_params["query"] == "machine learning"
    # The "year" parameter should have been added.
    assert called_params["year"] == "2020"
    # The limit is set to min(limit, 100) so it should be 3.
    assert called_params["limit"] == 3
    # The fields should be a comma-separated string from the dummy config.
    assert called_params["fields"] == "paperId,title,authors"


def test_search_tool_unexpected_format(monkeypatch):
    """
    Test that search_tool raises a RuntimeError when the API response
    does not include the expected 'data' key.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_no_data)
    tool_call_id = "test_tool_call_id"
    with pytest.raises(
        RuntimeError,
        match="Unexpected response from Semantic Scholar API. Please retry the same query.",
    ):
        search_tool.run(
            {
                "query": "test",
                "tool_call_id": tool_call_id,
            }
        )


def test_search_tool_no_papers(monkeypatch):
    """
    Test that search_tool raises a RuntimeError when the API returns no papers.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_no_papers)
    tool_call_id = "test_tool_call_id"
    with pytest.raises(
        RuntimeError,
        match="No papers returned from Semantic Scholar API. Please retry the same query.",
    ):
        search_tool.run(
            {
                "query": "test",
                "tool_call_id": tool_call_id,
            }
        )


def test_search_tool_requests_exception(monkeypatch):
    """
    Test that search_tool raises a RuntimeError when requests.get throws an exception.
    """
    monkeypatch.setattr(requests, "get", dummy_requests_get_exception)
    tool_call_id = "test_tool_call_id"
    with pytest.raises(
        RuntimeError,
        match="Failed to connect to Semantic Scholar API. Please retry the same query.",
    ):
        search_tool.run(
            {
                "query": "test",
                "tool_call_id": tool_call_id,
            }
        )
