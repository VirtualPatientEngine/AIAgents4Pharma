"""
Unit tests for S2 tools functionality.
"""

# pylint: disable=redefined-outer-name
from unittest.mock import patch
from unittest.mock import MagicMock
import pytest
from ..tools.s2.query_dataframe import query_dataframe, NoPapersFoundError


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

    def test_query_dataframe_empty_state(self, initial_state):
        """Tests query_dataframe tool behavior when no papers are found."""
        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first.",
        ):
            query_dataframe.invoke(
                {"question": "List all papers", "state": initial_state}
            )

    @patch(
        "aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent"
    )
    def test_query_dataframe_with_papers(self, mock_create_agent, initial_state):
        """Tests querying papers when data is available."""
        state = initial_state.copy()
        state["last_displayed_papers"] = "papers"
        state["papers"] = MOCK_STATE_PAPER

        # Mock the dataframe agent instead of the LLM
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Mocked response"}

        mock_create_agent.return_value = (
            mock_agent  # Mock the function returning the agent
        )

        # Ensure that the output of query_dataframe is correctly structured
        result = query_dataframe.invoke({"question": "List all papers", "state": state})

        assert isinstance(result, str)  # Ensure output is a string
        assert result == "Mocked response"  # Validate the expected response
