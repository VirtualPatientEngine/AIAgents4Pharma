"""
Unit tests for S2 tools functionality.
"""

# pylint: disable=redefined-outer-name
import pytest
from langgraph.types import Command
from ..tools.s2.display_dataframe import (
    display_dataframe,
    NoPapersFoundError as raised_error,
)


@pytest.fixture
def initial_state():
    """Provides an empty initial state for tests."""
    return {"papers": {}, "multi_papers": {}}


# Fixed test data for deterministic results
MOCK_SEARCH_RESPONSE = {
    "data": [
        {
            "paperId": "123",
            "title": "Machine Learning Basics",
            "abstract": "An introduction to ML",
            "year": 2023,
            "citationCount": 100,
            "url": "https://example.com/paper1",
            "authors": [{"name": "Test Author"}],
        }
    ]
}

MOCK_STATE_PAPER = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "URL": "https://example.com/paper1",
    }
}


class TestS2Tools:
    """Unit tests for individual S2 tools"""

    def test_display_dataframe_empty_state(self, initial_state):
        """Verifies display_dataframe tool behavior when state is empty and raises an exception"""
        with pytest.raises(
            raised_error,
            match="No papers found. A search/rec needs to be performed first.",
        ):
            display_dataframe.invoke({"state": initial_state, "tool_call_id": "test123"})

    def test_display_dataframe_shows_papers(self, initial_state):
        """Verifies display_dataframe tool correctly returns papers from state"""
        state = initial_state.copy()
        state["last_displayed_papers"] = "papers"
        state["papers"] = MOCK_STATE_PAPER

        result = display_dataframe.invoke(
            input={"state": state, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)  # Expect a Command object
        assert isinstance(result.update, dict)  # Ensure update is a dictionary
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert (
            "1 papers found. Papers are attached as an artifact."
            in result.update["messages"][0].content
        )
