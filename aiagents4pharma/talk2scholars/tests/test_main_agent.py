"""
Unit tests for main agent functionality.
Tests the supervisor agent's routing logic and state management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.main_agent import make_supervisor_node
from ..agents.main_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars
import pytest
from unittest.mock import Mock
from ..agents.main_agent import get_app


def test_get_app():
    """Test the full initialization of the LangGraph application."""
    thread_id = "test_thread"
    llm_model = "gpt-4o-mini"

    # Mock the LLM
    mock_llm = Mock()

    with patch('aiagents4pharma.talk2scholars.agents.main_agent.ChatOpenAI', return_value=mock_llm):
        app = get_app(thread_id, llm_model)
        assert app is not None


def test_supervisor_node_execution():
    """Test that the supervisor node processes messages and makes a decision."""
    mock_llm = Mock()
    thread_id = "test_thread"

    # Mock the supervisor agent's response
    mock_supervisor = Mock()
    mock_supervisor.invoke.return_value = {"messages": [AIMessage(content="s2_agent")]}

    with patch('aiagents4pharma.talk2scholars.agents.main_agent.create_react_agent', return_value=mock_supervisor):
        supervisor_node = make_supervisor_node(mock_llm, None, thread_id)

        # Create a mock state
        mock_state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])

        # Execute
        result = supervisor_node(mock_state)

        # Validate
        assert result.goto == "s2_agent"
        assert "messages" in result.update

def test_call_s2_agent():
    """Test the call to S2 agent and its integration with the state."""
    thread_id = "test_thread"
    mock_state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])

    with patch('aiagents4pharma.talk2scholars.agents.s2_agent.get_app') as mock_s2_app:
        mock_s2_app.return_value.invoke.return_value = {
            "messages": [AIMessage(content="Here are the papers")],
            "papers": {"id123": "Sample Paper"},
        }

        app = get_app(thread_id)
        result = app.invoke(
            mock_state, 
            config={  # ✅ Passing as config instead of kwargs
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint"
                }
            }
        )

        assert "messages" in result
        assert "papers" in result
        assert result["papers"]["id123"] == "Sample Paper"



def test_hydra_failure():
    """Test exception handling when Hydra fails to load config."""
    thread_id = "test_thread"
    with patch('hydra.initialize', side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)

        assert "Hydra error" in str(exc_info.value)


@pytest.fixture(autouse=True)
def mock_hydra():
    """Mock Hydra configuration."""
    with patch('hydra.initialize'), patch('hydra.compose') as mock_compose:
        cfg_mock = MagicMock()
        cfg_mock.agents.talk2scholars.main_agent.temperature = 0
        cfg_mock.agents.talk2scholars.main_agent.main_agent = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


class TestMainAgent:
    """Basic tests for the main agent initialization and configuration"""

    def test_supervisor_node_creation(self, mock_hydra):
        """Test that supervisor node can be created with correct config"""
        mock_llm = Mock()
        thread_id = "test_thread"
        
        with patch('aiagents4pharma.talk2scholars.agents.main_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            supervisor = make_supervisor_node(mock_llm, None, thread_id)
            
            assert supervisor is not None
            assert mock_create.called
            # Verify Hydra was called with correct parameters
            mock_hydra.assert_called_once()

    def test_supervisor_config_loading(self, mock_hydra):
        """Test that supervisor loads configuration correctly"""
        mock_llm = Mock()
        thread_id = "test_thread"
        
        with patch('aiagents4pharma.talk2scholars.agents.main_agent.create_react_agent'):
            make_supervisor_node(mock_llm, None, thread_id)
            
            # Verify Hydra initialization
            assert mock_hydra.call_count == 1
            assert "agents/talk2scholars/main_agent=default" in str(mock_hydra.call_args_list[0])

    def test_react_agent_params(self):
        """Test that react agent is created with correct parameters"""
        mock_llm = Mock()
        thread_id = "test_thread"
        
        with patch('aiagents4pharma.talk2scholars.agents.main_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            make_supervisor_node(mock_llm, None, thread_id)
            
            # Verify create_react_agent was called
            assert mock_create.called
            
            # Verify the parameters
            args, kwargs = mock_create.call_args
            assert args[0] == mock_llm  # First argument should be the LLM
            assert "state_schema" in kwargs  # Should have state_schema
            assert hasattr(mock_create.return_value, 'invoke')  # Should have invoke method

    def test_supervisor_custom_config(self, mock_hydra):
        """Test supervisor with custom configuration"""
        mock_llm = Mock()
        custom_config = {"custom_param": "test_value"}
        thread_id = "test_thread"
        
        with patch('aiagents4pharma.talk2scholars.agents.main_agent.create_react_agent'):
            make_supervisor_node(mock_llm, custom_config, thread_id)
            
            # Verify Hydra was called
            mock_hydra.assert_called_once()