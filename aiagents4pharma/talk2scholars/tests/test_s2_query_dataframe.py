"""
Unit tests for S2 tools functionality.
"""

# pylint: disable=redefined-outer-name
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from ..tools.s2.query_dataframe import NoPapersFoundError, query_dataframe


@pytest.fixture
def initial_state():
    """Provides an empty initial state for tests with a dummy llm_model."""

    return {"papers": {}, "multi_papers": {}, "llm_model": MagicMock()}


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
        # Calling without any papers should raise NoPapersFoundError
        tool_input = {
            "question": "List all papers",
            "state": initial_state,
            "tool_call_id": "test_id",
        }
        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first.",
        ):
            query_dataframe.run(tool_input)

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
        # Invoke the tool with a test tool_call_id
        tool_input = {
            "question": "List all papers",
            "state": state,
            "tool_call_id": "test_id",
        }
        result = query_dataframe.run(tool_input)
        # The tool returns a Command with messages
        assert hasattr(result, "update")
        update = result.update
        assert "messages" in update
        msgs = update["messages"]
        assert len(msgs) == 1
        msg = msgs[0]
        assert isinstance(msg, ToolMessage)
        assert msg.content == "Mocked response"

    @patch(
        "aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent"
    )
    def test_query_dataframe_direct_mapping(self, mock_create_agent, initial_state):
        """Tests query_dataframe when last_displayed_papers is a direct dict mapping."""
        # Prepare state with direct mapping
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPER
        # Mock the dataframe agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Direct mapping response"}
        mock_create_agent.return_value = mock_agent
        # Invoke tool
        # Invoke the tool with direct mapping and test tool_call_id
        tool_input = {
            "question": "Filter papers",
            "state": state,
            "tool_call_id": "test_id",
        }
        result = query_dataframe.run(tool_input)
        update = result.update
        assert "messages" in update
        msgs = update["messages"]
        assert len(msgs) == 1
        msg = msgs[0]
        assert isinstance(msg, ToolMessage)
        assert msg.content == "Direct mapping response"

    def test_query_dataframe_missing_llm(self, initial_state):
        """Test that missing llm_model raises ValueError."""
        # Remove llm_model
        state = {k: v for k, v in initial_state.items() if k != "llm_model"}
        state["last_displayed_papers"] = MOCK_STATE_PAPER
        tool_input = {"question": "Test", "state": state, "tool_call_id": "test_id"}
        with pytest.raises(ValueError) as exc:
            query_dataframe.run(tool_input)
        assert "Missing 'llm_model' in state." in str(exc.value)

    def test_query_dataframe_invalid_mapping(self, initial_state):
        """Test that invalid last_displayed_papers mapping raises ValueError."""
        # Provide invalid mapping key
        state = initial_state.copy()
        state["last_displayed_papers"] = "nonexistent_key"
        # llm_model present
        tool_input = {"question": "Test", "state": state, "tool_call_id": "test_id"}
        with pytest.raises(ValueError) as exc:
            query_dataframe.run(tool_input)
        assert "Could not resolve a valid metadata dictionary" in str(exc.value)
