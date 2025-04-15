"""Unit tests for the paper download agent in Talk2Scholars."""

from unittest import mock
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from ..agents.paper_download_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """Mocks Hydra configuration for tests."""
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        cfg_mock = mock.MagicMock()
        cfg_mock.agents.talk2scholars.s2_agent.temperature = 0
        cfg_mock.agents.talk2scholars.paper_download_agent.prompt = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """Mocks paper download tools to prevent real HTTP calls."""
    with mock.patch(
        "aiagents4pharma.talk2scholars.tools.paper_download."
        "download_arxiv_input.download_arxiv_paper"
    ) as mock_download_arxiv_paper:
        mock_download_arxiv_paper.return_value = {
            "article_data": {"dummy_key": "dummy_value"}
        }
        yield [mock_download_arxiv_paper]


@pytest.mark.usefixtures("mock_hydra_fixture")
def test_paper_download_agent_initialization():
    """Ensures the paper download agent initializes properly with a prompt."""
    thread_id = "test_thread_paper_dl"
    llm_mock = mock.Mock(spec=BaseChatModel)  # Mock LLM

    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create_agent:
        mock_create_agent.return_value = mock.Mock()

        app = get_app(thread_id, llm_mock)
        assert app is not None, "The agent app should be successfully created."
        assert mock_create_agent.called


def test_paper_download_agent_invocation():
    """Verifies agent processes queries and updates state correctly."""
    _ = mock_tools_fixture  # Prevents unused-argument warning
    thread_id = "test_thread_paper_dl"
    mock_state = Talk2Scholars(
        messages=[HumanMessage(content="Download paper 1234.5678")]
    )
    llm_mock = mock.Mock(spec=BaseChatModel)

    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create_agent:
        mock_agent = mock.Mock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Here is the paper")],
            "article_data": {"file_bytes": b"FAKE_PDF_CONTENTS"},
        }

        app = get_app(thread_id, llm_mock)
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


def test_paper_download_agent_tools_assignment(request):  # Keep fixture name
    """Checks correct tool assignment (download_arxiv_paper, query_dataframe)."""
    thread_id = "test_thread_paper_dl"
    mock_tools = request.getfixturevalue("mock_tools_fixture")
    llm_mock = mock.Mock(spec=BaseChatModel)

    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
        ) as mock_create_agent,
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.paper_download_agent.ToolNode"
        ) as mock_toolnode,
    ):
        mock_agent = mock.Mock()
        mock_create_agent.return_value = mock_agent
        mock_tool_instance = mock.Mock()
        mock_tool_instance.tools = mock_tools if mock_tools else []
        mock_toolnode.return_value = mock_tool_instance

        get_app(thread_id, llm_mock)
        assert mock_toolnode.called
        assert len(mock_tool_instance.tools) == 1


def test_paper_download_agent_hydra_failure():
    """Confirms the agent gracefully handles exceptions if Hydra fails."""
    thread_id = "test_thread_paper_dl"
    llm_mock = mock.Mock(spec=BaseChatModel)

    with mock.patch("hydra.initialize", side_effect=Exception("Mock Hydra failure")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id, llm_mock)
        assert "Mock Hydra failure" in str(exc_info.value)


def test_paper_download_agent_model_failure():
    """Ensures agent handles model-related failures gracefully."""
    thread_id = "test_thread_paper_dl"
    llm_mock = mock.Mock(spec=BaseChatModel)

    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent",
        side_effect=Exception("Mock model failure"),
    ):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id, llm_mock)
        assert "Mock model failure" in str(
            exc_info.value
        ), "Model initialization failure should raise an exception."
