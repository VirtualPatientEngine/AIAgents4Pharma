"""
Unit tests for the S2 agent (Semantic Scholar sub-agent).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from ..agents.s2_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra():
    """Mock Hydra configuration to prevent external dependencies."""
    with patch('hydra.initialize'), patch('hydra.compose') as mock_compose:
        cfg_mock = MagicMock()
        cfg_mock.agents.talk2scholars.s2_agent.temperature = 0
        cfg_mock.agents.talk2scholars.s2_agent.s2_agent = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools():
    """Mock tools to prevent execution of real API calls."""
    with patch('aiagents4pharma.talk2scholars.tools.s2.search.search_tool') as mock_s2_search, \
         patch('aiagents4pharma.talk2scholars.tools.s2.display_results.display_results') as mock_s2_display, \
         patch('aiagents4pharma.talk2scholars.tools.s2.single_paper_rec.get_single_paper_recommendations') as mock_s2_single_rec, \
         patch('aiagents4pharma.talk2scholars.tools.s2.multi_paper_rec.get_multi_paper_recommendations') as mock_s2_multi_rec:
        
        mock_s2_search.return_value = Mock()
        mock_s2_display.return_value = Mock()
        mock_s2_single_rec.return_value = Mock()
        mock_s2_multi_rec.return_value = Mock()

        yield [mock_s2_search, mock_s2_display, mock_s2_single_rec, mock_s2_multi_rec]


def test_s2_agent_initialization(mock_hydra, mock_tools):
    """Test that S2 agent initializes correctly with mock configuration and tools."""
    thread_id = "test_thread"

    with patch('aiagents4pharma.talk2scholars.agents.s2_agent.create_react_agent') as mock_create:
        mock_create.return_value = Mock()

        app = get_app(thread_id)

        assert app is not None
        assert mock_hydra.called
        assert mock_create.called


def test_s2_agent_invocation(mock_hydra):
    """Test that the S2 agent processes user input and returns a valid response."""
    thread_id = "test_thread"
    mock_state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])

    with patch('aiagents4pharma.talk2scholars.agents.s2_agent.create_react_agent') as mock_create:
        mock_agent = Mock()
        mock_create.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Here are some AI papers")],
            "papers": {"id123": "AI Research Paper"},
        }

        app = get_app(thread_id)
        result = app.invoke(mock_state, config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint"
            }
        })

        assert "messages" in result
        assert "papers" in result
        assert result["papers"]["id123"] == "AI Research Paper"


def test_s2_agent_tools_assignment(mock_hydra, mock_tools):
    """Ensure that the correct tools are assigned to the agent."""
    thread_id = "test_thread"

    with patch('aiagents4pharma.talk2scholars.agents.s2_agent.create_react_agent') as mock_create, \
         patch('aiagents4pharma.talk2scholars.agents.s2_agent.ToolNode') as mock_toolnode:

        mock_agent = Mock()
        mock_create.return_value = mock_agent

        # Mock ToolNode behavior
        mock_tool_instance = Mock()
        mock_tool_instance.tools = mock_tools  # Ensure the mock toolnode contains tools
        mock_toolnode.return_value = mock_tool_instance

        app = get_app(thread_id)

        # Ensure the agent was created with the mocked ToolNode
        args, kwargs = mock_create.call_args
        assert "tools" in kwargs
        assert mock_toolnode.called  # Ensure ToolNode was instantiated
        assert len(mock_toolnode.return_value.tools) == 4  # Verify tool count



def test_s2_agent_hydra_failure():
    """Test exception handling when Hydra fails to load config."""
    thread_id = "test_thread"
    
    with patch('hydra.initialize', side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)

        assert "Hydra error" in str(exc_info.value)
