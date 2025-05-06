"""
Unit Tests for the PDF agent.
"""

# pylint: disable=redefined-outer-name
from unittest import mock
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.pdf_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """Mock Hydra configuration to prevent external dependencies."""
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        # Create a mock configuration with a pdf_agent section.
        cfg_mock = mock.MagicMock()
        # The pdf_agent config will be accessed as cfg.agents.talk2scholars.pdf_agent in get_app.
        cfg_mock.agents.talk2scholars.pdf_agent.some_property = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """Mock PDF agent tools to prevent execution of real API calls."""
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.pdf_agent.question_and_answer"
        ) as mock_question_and_answer,
    ):
        mock_question_and_answer.return_value = {
            "result": "Mock Question and Answer Result"
        }
        yield [mock_question_and_answer]


@pytest.fixture
def mock_llm():
    """Provide a dummy language model to pass into get_app."""
    return mock.Mock()


@pytest.mark.usefixtures("mock_hydra_fixture")
def test_pdf_agent_initialization(mock_llm):
    """Test that PDF agent initializes correctly with mock configuration."""
    thread_id = "test_thread"
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent"
    ) as mock_create:
        mock_create.return_value = mock.Mock()
        app = get_app(thread_id, mock_llm)
        assert app is not None
        assert mock_create.called


def test_pdf_agent_invocation(mock_llm):
    """Test that the PDF agent processes user input and returns a valid response."""
    thread_id = "test_thread"
    # Create a sample state with a human message.
    mock_state = Talk2Scholars(
        messages=[HumanMessage(content="Extract key data from PDF")]
    )
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent"
    ) as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        # Simulate a response from the PDF agent.
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="PDF content extracted successfully")],
            "article_data": {"page": 1, "text": "Sample PDF text"},
        }
        app = get_app(thread_id, mock_llm)
        result = app.invoke(
            mock_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )
        assert "messages" in result
        assert "article_data" in result
        assert result["article_data"]["page"] == 1


def test_pdf_agent_tools_assignment(request, mock_llm):
    """Ensure that the correct tools are assigned to the PDF agent."""
    thread_id = "test_thread"
    mock_tools = request.getfixturevalue("mock_tools_fixture")
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent"
        ) as mock_create,
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.pdf_agent.ToolNode"
        ) as mock_toolnode,
    ):
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        mock_tool_instance = mock.Mock()
        mock_tool_instance.tools = mock_tools
        mock_toolnode.return_value = mock_tool_instance
        get_app(thread_id, mock_llm)
        assert mock_toolnode.called
        assert len(mock_tool_instance.tools) == 1


def test_pdf_agent_hydra_failure(mock_llm):
    """Test exception handling when Hydra fails to load config for PDF agent."""
    thread_id = "test_thread"
    with mock.patch("hydra.initialize", side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id, mock_llm)
        assert "Hydra error" in str(exc_info.value)
