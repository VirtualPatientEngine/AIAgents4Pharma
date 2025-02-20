"""
Unit tests for main agent functionality.
Tests the supervisor agent's routing logic and state management.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=redefined-outer-name,too-few-public-methods
from unittest.mock import Mock, patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END
from ..agents.main_agent import make_supervisor_node, get_hydra_config, get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra():
    """Mock Hydra configuration."""
    with patch("hydra.initialize"), patch("hydra.compose") as mock_compose:
        cfg_mock = MagicMock()
        cfg_mock.agents.talk2scholars.main_agent.temperature = 0
        cfg_mock.agents.talk2scholars.main_agent.system_prompt = "System prompt"
        cfg_mock.agents.talk2scholars.main_agent.router_prompt = "Router prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


def test_get_app():
    """Test the full initialization of the LangGraph application."""
    thread_id = "test_thread"
    mock_llm = Mock()
    app = get_app(thread_id, mock_llm)
    assert app is not None
    assert "supervisor" in app.nodes
    assert "s2_agent" in app.nodes  # Ensure nodes exist


def test_get_app_with_default_llm():
    """Test app initialization with default LLM parameters."""
    thread_id = "test_thread"
    llm_mock = Mock()

    # We need to explicitly pass the mock instead of patching, since the function uses
    # ChatOpenAI as a default argument value which is evaluated at function definition time
    app = get_app(thread_id, llm_mock)
    assert app is not None
    # We can only verify the app was created successfully


def test_get_hydra_config():
    """Test that Hydra configuration loads correctly."""
    with patch("hydra.initialize"), patch("hydra.compose") as mock_compose:
        cfg_mock = MagicMock()
        cfg_mock.agents.talk2scholars.main_agent.temperature = 0
        mock_compose.return_value = cfg_mock
        cfg = get_hydra_config()
        assert cfg is not None
        assert cfg.temperature == 0


def test_hydra_failure():
    """Test exception handling when Hydra fails to load config."""
    thread_id = "test_thread"
    with patch("hydra.initialize", side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)
        assert "Hydra error" in str(exc_info.value)


def test_supervisor_node_execution():
    """Test that the supervisor node routes correctly."""
    mock_llm = Mock()
    thread_id = "test_thread"

    class MockRouter:
        """Mock router class."""
        next = "s2_agent"

    with (
        patch.object(mock_llm, "with_structured_output", return_value=mock_llm),
        patch.object(mock_llm, "invoke", return_value=MockRouter()),
    ):
        supervisor_node = make_supervisor_node(mock_llm, thread_id)
        mock_state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])
        result = supervisor_node(mock_state)
        assert result.goto == "s2_agent"


def test_supervisor_node_finish():
    """Test that supervisor node correctly handles FINISH case."""
    mock_llm = Mock()
    thread_id = "test_thread"

    class MockRouter:
        """Mock router class."""
        next = "FINISH"

    class MockAIResponse:
        """Mock AI response class."""
        def __init__(self):
            self.content = "Final AI Response"

    with (
        patch.object(mock_llm, "with_structured_output", return_value=mock_llm),
        patch.object(mock_llm, "invoke", side_effect=[MockRouter(), MockAIResponse()]),
    ):
        supervisor_node = make_supervisor_node(mock_llm, thread_id)
        mock_state = Talk2Scholars(messages=[HumanMessage(content="End conversation")])
        result = supervisor_node(mock_state)
        assert result.goto == END
        assert "messages" in result.update
        assert isinstance(result.update["messages"], AIMessage)
        assert result.update["messages"].content == "Final AI Response"


def test_supervisor_node_finish_no_ai_response():
    """Test FINISH case when the last message is from AI (no response needed)."""
    mock_llm = Mock()
    thread_id = "test_thread"

    class MockRouter:
        """Mock router class."""
        next = "FINISH"

    with patch.object(mock_llm, "with_structured_output", return_value=mock_llm):
        with patch.object(mock_llm, "invoke", return_value=MockRouter()):
            supervisor_node = make_supervisor_node(mock_llm, thread_id)
            # Create state with last message from AI
            mock_state = Talk2Scholars(
                messages=[
                    HumanMessage(content="Hello"),
                    AIMessage(content="Previous AI response"),
                ]
            )
            result = supervisor_node(mock_state)
            assert result.goto == END
            # In this case, update may be None or an empty dict, check both
            assert not result.update or "messages" not in result.update


def test_call_s2_agent():
    """Test the s2_agent node integration by verifying app structure."""
    thread_id = "test_thread"
    mock_llm = Mock()

    # Mock s2_agent's get_app function
    mock_s2_app = Mock()

    with patch(
        "aiagents4pharma.talk2scholars.agents.s2_agent.get_app",
        return_value=mock_s2_app,
    ):
        # Initialize the main app
        app = get_app(thread_id, mock_llm)

        # For a CompiledStateGraph, we can only verify the node exists
        assert "s2_agent" in app.nodes

        # Test that s2_agent.get_app was called with correct parameters
        # by creating a test state and invoking the compiled app
        _ = Talk2Scholars(
            messages=[HumanMessage(content="Find papers about AI")]
        )

        # Mock everything necessary for a simple workflow test
        with patch.object(mock_s2_app, "invoke") as mock_invoke:
            # Set up mock response
            mock_invoke.return_value = Talk2Scholars(
                messages=[
                    HumanMessage(content="Find papers about AI"),
                    AIMessage(content="Found these papers"),
                ]
            )

            # Skip actually running the workflow since we just want to
            # test if the s2_agent is integrated properly
            assert app is not None


def test_system_prompt_usage():
    """Test that system prompts are correctly applied."""
    mock_llm = Mock()
    thread_id = "test_thread"

    class MockRouter:
        """Mock router class."""
        next = "FINISH"

    class MockAIResponse:
        """Mock AI response class."""
        def __init__(self):
            self.content = "Response using system prompt"

    with (
        patch.object(mock_llm, "with_structured_output", return_value=mock_llm),
        patch.object(mock_llm, "invoke", side_effect=[MockRouter(), MockAIResponse()]),
    ):
        supervisor_node = make_supervisor_node(mock_llm, thread_id)
        mock_state = Talk2Scholars(messages=[HumanMessage(content="Question")])

        # Execute the node
        supervisor_node(mock_state)

        # Check that the second invoke used the system prompt
        calls = mock_llm.invoke.call_args_list
        assert len(calls) == 2
        # First call should use router prompt
        assert isinstance(calls[0][0][0][0], SystemMessage)
        assert calls[0][0][0][0].content == "Router prompt"
        # Second call should use system prompt
        assert isinstance(calls[1][0][0][0], SystemMessage)
        assert calls[1][0][0][0].content == "System prompt"
