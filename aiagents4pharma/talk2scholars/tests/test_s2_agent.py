"""
Unit tests for the S2 agent (Semantic Scholar sub-agent).
"""

from unittest import mock
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.s2_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """Mock Hydra configuration to prevent external dependencies."""
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        cfg_mock = mock.MagicMock()
        cfg_mock.agents.talk2scholars.s2_agent.temperature = 0
        cfg_mock.agents.talk2scholars.s2_agent.s2_agent = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """Mock tools to prevent execution of real API calls."""
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.search.search_tool"
        ) as mock_s2_search,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.display_results.display_results"
        ) as mock_s2_display,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.single_paper_rec."
            "get_single_paper_recommendations"
        ) as mock_s2_single_rec,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.multi_paper_rec."
            "get_multi_paper_recommendations"
        ) as mock_s2_multi_rec,
    ):

        mock_s2_search.return_value = mock.Mock()
        mock_s2_display.return_value = mock.Mock()
        mock_s2_single_rec.return_value = mock.Mock()
        mock_s2_multi_rec.return_value = mock.Mock()

        yield [mock_s2_search, mock_s2_display, mock_s2_single_rec, mock_s2_multi_rec]


@pytest.mark.usefixtures("mock_hydra_fixture")
def test_s2_agent_initialization():
    """Test that S2 agent initializes correctly with mock configuration."""
    thread_id = "test_thread"

    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.s2_agent.create_react_agent"
    ) as mock_create:
        mock_create.return_value = mock.Mock()

        app = get_app(thread_id)

        assert app is not None
        assert mock_create.called


def test_s2_agent_invocation():
    """Test that the S2 agent processes user input and returns a valid response."""
    thread_id = "test_thread"
    mock_state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])

    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.s2_agent.create_react_agent"
    ) as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Here are some AI papers")],
            "papers": {"id123": "AI Research Paper"},
        }

        app = get_app(thread_id)
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
        assert "papers" in result
        assert result["papers"]["id123"] == "AI Research Paper"


def test_s2_agent_tools_assignment(request):
    """Ensure that the correct tools are assigned to the agent."""
    thread_id = "test_thread"

    # Dynamically retrieve the fixture to avoid redefining it in the function signature
    mock_tools = request.getfixturevalue("mock_tools_fixture")

    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.s2_agent.create_react_agent"
        ) as mock_create,
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.s2_agent.ToolNode"
        ) as mock_toolnode,
    ):

        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent

        # Mock ToolNode behavior
        mock_tool_instance = mock.Mock()
        mock_tool_instance.tools = mock_tools  # Use the dynamically retrieved fixture
        mock_toolnode.return_value = mock_tool_instance

        get_app(thread_id)

        # Ensure the agent was created with the mocked ToolNode
        assert mock_toolnode.called
        assert len(mock_tool_instance.tools) == 4


def test_s2_agent_hydra_failure():
    """Test exception handling when Hydra fails to load config."""
    thread_id = "test_thread"

    with mock.patch("hydra.initialize", side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)

        assert "Hydra error" in str(exc_info.value)
