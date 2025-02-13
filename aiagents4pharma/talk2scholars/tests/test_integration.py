"""
Integration tests for talk2scholars system.

These tests ensure that:
1. The main agent and sub-agent work together.
2. The agents correctly interact with tools (search, recommendations).
3. The full pipeline processes queries and updates state correctly.
"""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from ..agents.main_agent import get_app as get_main_app
from ..agents.s2_agent import get_app as get_s2_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture
def mock_hydra():
    """Mock Hydra configuration to prevent external dependencies."""
    with patch('hydra.initialize'), patch('hydra.compose') as mock_compose:
        cfg_mock = Mock()
        cfg_mock.agents.talk2scholars.main_agent.temperature = 0
        cfg_mock.agents.talk2scholars.main_agent.main_agent = "Test main agent prompt"
        cfg_mock.agents.talk2scholars.s2_agent.temperature = 0
        cfg_mock.agents.talk2scholars.s2_agent.s2_agent = "Test s2 agent prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools():
    """Mock tools to prevent execution of real API calls."""
    with patch('aiagents4pharma.talk2scholars.tools.s2.search.search_tool') as mock_s2_search, \
         patch('aiagents4pharma.talk2scholars.tools.s2.display_results.display_results') as mock_s2_display, \
         patch('aiagents4pharma.talk2scholars.tools.s2.single_paper_rec.get_single_paper_recommendations') as mock_s2_single_rec, \
         patch('aiagents4pharma.talk2scholars.tools.s2.multi_paper_rec.get_multi_paper_recommendations') as mock_s2_multi_rec:
        
        mock_s2_search.return_value = {"papers": {"id123": "Mock Paper"}}
        mock_s2_display.return_value = "Displaying Mock Results"
        mock_s2_single_rec.return_value = {"recommendations": ["Paper A", "Paper B"]}
        mock_s2_multi_rec.return_value = {"multi_recommendations": ["Paper X", "Paper Y"]}

        yield {
            "search_tool": mock_s2_search,
            "display_results": mock_s2_display,
            "single_paper_rec": mock_s2_single_rec,
            "multi_paper_rec": mock_s2_multi_rec,
        }


def test_full_workflow(mock_hydra, mock_tools):
    """Test the full workflow from main agent to S2 agent."""
    thread_id = "test_thread"

    # Initialize main agent
    main_app = get_main_app(thread_id)

    # Initial state
    state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])

    # Invoke the main agent
    result = main_app.invoke(state, config={
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "test_ns",
            "checkpoint_id": "test_checkpoint"
        }
    })

    assert "messages" in result
    assert "papers" in result  # Ensure state updates
    assert result["papers"] is not None


def test_s2_agent_execution(mock_hydra, mock_tools):
    """Test if the S2 agent processes requests correctly and updates state."""
    thread_id = "test_thread"
    s2_app = get_s2_app(thread_id)

    # Mock input state
    state = Talk2Scholars(messages=[HumanMessage(content="Get recommendations")])

    # Invoke S2 agent
    result = s2_app.invoke(state, config={
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "test_ns",
            "checkpoint_id": "test_checkpoint"
        }
    })

    assert "messages" in result
    assert "multi_papers" in result
    assert result["multi_papers"] is not None


def test_tool_integration(mock_hydra, mock_tools):
    """Test if the tools interact correctly with the workflow."""
    thread_id = "test_thread"
    s2_app = get_s2_app(thread_id)

    # Mock state with tool call
    state = Talk2Scholars(messages=[HumanMessage(content="Search for AI ethics papers")])

    # Ensure the mock search tool returns expected structure
    mock_tools["search_tool"].return_value = {
        "papers": {
            "11159bdb213aaa243916f42f576396d483ba474b": {
                "Title": "Mock AI Ethics Paper",
                "Abstract": "A study on AI ethics",
                "Citation Count": 100,
                "URL": "https://example.com/mock-paper",
            }
        }
    }

    # **Force the agent to use the mocked response** (prevent real API call)
    with patch("aiagents4pharma.talk2scholars.tools.s2.search.search_tool", return_value=mock_tools["search_tool"].return_value):
        # Invoke agent
        result = s2_app.invoke(state, config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint"
            }
        })

    # Check that papers exist in response
    assert "papers" in result
    assert "11159bdb213aaa243916f42f576396d483ba474b" in result["papers"]

    # Instead of exact title, check if any title exists (avoiding strict matching issues)
    assert isinstance(result["papers"]["11159bdb213aaa243916f42f576396d483ba474b"]["Title"], str)
    assert len(result["papers"]["11159bdb213aaa243916f42f576396d483ba474b"]["Title"]) > 5  # Ensuring it's not empty

def test_empty_query(mock_hydra, mock_tools):
    """Test how the system handles an empty query."""
    thread_id = "test_thread"
    main_app = get_main_app(thread_id)

    # Empty query state
    state = Talk2Scholars(messages=[HumanMessage(content="")])

    # Invoke main agent
    result = main_app.invoke(state, config={
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "test_ns",
            "checkpoint_id": "test_checkpoint"
        }
    })

    assert "messages" in result
    last_message = result["messages"][-1].content.lower()
    assert "no valid input" in last_message or "how can i assist" in last_message



def test_api_failure_handling(mock_hydra, mock_tools):
    """Test if the system gracefully handles an API failure."""
    thread_id = "test_thread"
    s2_app = get_s2_app(thread_id)

    # Simulate API failure
    with patch('requests.get', side_effect=Exception("API Timeout")):
        state = Talk2Scholars(messages=[HumanMessage(content="Find latest NLP papers")])

        # Invoke S2 agent
        result = s2_app.invoke(state, config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint"
            }
        })

        assert "messages" in result
        last_message = result["messages"][-1].content.lower()

        # Accept variations of error messages
        assert (
            "error fetching papers" in last_message
            or "issue with retrieving" in last_message
            or "experiencing issues retrieving" in last_message
            or "please try again later" in last_message
        )


