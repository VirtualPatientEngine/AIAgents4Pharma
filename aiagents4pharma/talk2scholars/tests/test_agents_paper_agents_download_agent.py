"""Unit tests for the paper download agent in Talk2Scholars."""

from unittest import mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ..agents.paper_download_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """Mocks Hydra configuration for tests."""
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        cfg_mock = mock.MagicMock()
        cfg_mock.agents.talk2scholars.paper_download_agent.paper_download_agent = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """Mocks paper download tools to prevent real HTTP calls."""
    with mock.patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.download_papers"
    ) as mock_download_papers:
        mock_download_papers.return_value = {"article_data": {"dummy_key": "dummy_value"}}
        yield [mock_download_papers]


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
    mock_state = Talk2Scholars(messages=[HumanMessage(content="Download paper 1234.5678")])
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


def test_paper_download_agent_tools_assignment(
    request,
):
    """Checks correct tool assignment (download_papers tool)."""
    thread_id = "test_thread_paper_dl"
    request.getfixturevalue("mock_tools_fixture")
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
        mock_toolnode.return_value = mock_tool_instance

        get_app(thread_id, llm_mock)
        # Verify ToolNode was called with download_papers function
        assert mock_toolnode.called
        # Check that ToolNode was called with a list containing the download_papers tool
        call_args = mock_toolnode.call_args[0][0]  # Get first positional argument (the tools list)
        assert len(call_args) == 1
        # The tool should be a StructuredTool with name 'download_papers'
        tool = call_args[0]
        assert hasattr(tool, "name")
        assert tool.name == "download_papers"


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
        assert "Mock model failure" in str(exc_info.value), (
            "Model initialization failure should raise an exception."
        )
