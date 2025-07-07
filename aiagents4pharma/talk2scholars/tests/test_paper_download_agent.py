"""Unit tests for the paper download agent in Talk2Scholars."""

from unittest import mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aiagents4pharma.talk2scholars.agents.paper_download_agent import get_app
from aiagents4pharma.talk2scholars.state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """Mocks Hydra configuration for tests."""
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        cfg_mock = mock.MagicMock()
        cfg_mock.agents.talk2scholars.paper_download_agent.prompt = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """Mocks the paper_download.download_tool.download_paper function."""
    target = "aiagents4pharma.talk2scholars.tools.paper_download.download_tool.download_paper"
    with mock.patch(target) as mock_download_paper:
        mock_download_paper.return_value = {
            "article_data": {"dummy_key": "dummy_value"}
        }
        yield mock_download_paper


@pytest.mark.usefixtures("mock_hydra_fixture")
def test_paper_download_agent_initialization():
    """Ensures the paper download agent initializes properly with a prompt."""
    thread_id = "test_thread_paper_dl"
    llm_mock = mock.Mock(spec=BaseChatModel)

    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create_agent:
        mock_create_agent.return_value = mock.Mock()

        app = get_app(thread_id, llm_mock)
        assert app is not None
        assert mock_create_agent.called


def test_paper_download_agent_invocation(mock_tools_fixture):
    """Verifies agent processes queries and updates state correctly."""
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


def test_paper_download_agent_tools_assignment():
    """Checks correct tool assignment (download_paper only)."""
    thread_id = "test_thread_paper_dl"
    llm_mock = mock.Mock(spec=BaseChatModel)

    # Patch create_react_agent so we never actually build the model
    # Patch ToolNode so we can inspect what got passed in
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
        ),
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.paper_download_agent.ToolNode"
        ) as mock_toolnode,
    ):

        # When ToolNode(...) is called, return a dummy instance
        mock_toolnode.return_value = mock.Mock()

        # Call the factory
        get_app(thread_id, llm_mock)

        # Inspect the arguments passed to ToolNode
        assert mock_toolnode.call_count == 1, "ToolNode should be instantiated once"
        tools_list = mock_toolnode.call_args.args[0]

        # There should be exactly one tool
        assert isinstance(tools_list, list)
        assert len(tools_list) == 1

        tool = tools_list[0]
        # That tool must have the correct name
        assert (
            getattr(tool, "name", None) == "download_paper"
        ), f"Expected a tool named 'download_paper', got {tool!r}"
        # And a description, just to be sure it's the right StructuredTool
        assert hasattr(tool, "description") and isinstance(tool.description, str)


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
        assert "Mock model failure" in str(exc_info.value)
