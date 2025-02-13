"""
Integration tests for talk2scholars system.

These tests ensure that:
1. The main agent and sub-agent work together.
2. The agents correctly interact with tools (search, recommendations).
3. The full pipeline processes queries and updates state correctly.
"""
# pylint: disable=redefined-outer-name
from unittest.mock import patch, Mock
import pytest
from langchain_core.messages import HumanMessage

from ..agents.main_agent import get_app as get_main_app
from ..agents.s2_agent import get_app as get_s2_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
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


@pytest.fixture(autouse=True)
def mock_tools():
    """Mock tools to prevent execution of real API calls."""
    with patch(
        'aiagents4pharma.talk2scholars.tools.s2.search.search_tool'
    ) as mock_s2_search, patch(
        'aiagents4pharma.talk2scholars.tools.s2.display_results.display_results'
    ) as mock_s2_display, patch(
        'aiagents4pharma.talk2scholars.tools.s2.single_paper_rec.get_single_paper_recommendations'
    ) as mock_s2_single_rec, patch(
        'aiagents4pharma.talk2scholars.tools.s2.multi_paper_rec.get_multi_paper_recommendations'
    ) as mock_s2_multi_rec:

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


def test_full_workflow():
    """Test the full workflow from main agent to S2 agent."""
    thread_id = "test_thread"
    main_app = get_main_app(thread_id)

    state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])

    result = main_app.invoke(
        state,
        config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint",
            }
        },
    )

    assert "messages" in result
    assert "papers" in result
    assert result["papers"] is not None


def test_s2_agent_execution():
    """Test if the S2 agent processes requests correctly and updates state."""
    thread_id = "test_thread"
    s2_app = get_s2_app(thread_id)

    state = Talk2Scholars(messages=[HumanMessage(content="Get recommendations")])

    result = s2_app.invoke(
        state,
        config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint",
            }
        },
    )

    assert "messages" in result
    assert "multi_papers" in result
    assert result["multi_papers"] is not None


def test_tool_integration(mock_tools):
    """Test if the tools interact correctly with the workflow."""
    thread_id = "test_thread"
    s2_app = get_s2_app(thread_id)

    state = Talk2Scholars(messages=[HumanMessage(content="Search for AI ethics papers")])

    mock_paper_id = "11159bdb213aaa243916f42f576396d483ba474b"
    mock_response = {
        "papers": {
            mock_paper_id: {
                "Title": "Mock AI Ethics Paper",
                "Abstract": "A study on AI ethics",
                "Citation Count": 100,
                "URL": "https://example.com/mock-paper",
            }
        }
    }

    # Ensure both mocks return the same response
    mock_tools["search_tool"].return_value = mock_response

    with patch(
        'aiagents4pharma.talk2scholars.tools.s2.search.search_tool',
        return_value=mock_response,
    ):
        result = s2_app.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )

    assert "papers" in result
    assert mock_paper_id in result["papers"]



def test_empty_query():
    """Test how the system handles an empty query."""
    thread_id = "test_thread"
    main_app = get_main_app(thread_id)

    state = Talk2Scholars(messages=[HumanMessage(content="")])

    result = main_app.invoke(
        state,
        config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint",
            }
        },
    )

    assert "messages" in result
    last_message = result["messages"][-1].content.lower()
    assert "no valid input" in last_message or "how can i assist" in last_message


def test_api_failure_handling():
    """Test if the system gracefully handles an API failure."""
    thread_id = "test_thread"
    s2_app = get_s2_app(thread_id)

    with patch('requests.get', side_effect=Exception("API Timeout")):
        state = Talk2Scholars(messages=[HumanMessage(content="Find latest NLP papers")])

        result = s2_app.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )

        assert "messages" in result
        last_message = result["messages"][-1].content.lower()
        assert (
            "error fetching papers" in last_message
            or "issue with retrieving" in last_message
            or "experiencing issues retrieving" in last_message
            or "please try again later" in last_message
        )
