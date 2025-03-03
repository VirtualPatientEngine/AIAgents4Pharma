# File: tests/test_paper_download_agent.py

import pytest
from unittest import mock
from langchain_core.messages import HumanMessage, AIMessage

# Adjust these import paths as needed to match your project structure.
from ..agents.paper_download_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """
    Mocks Hydra initialization and composition to ensure
    we do not load any external configuration files during testing.
    """
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        # Simulate a minimal config structure required by the agent
        cfg_mock = mock.MagicMock()
        cfg_mock.agents.talk2scholars.paper_download_agent.paper_download_agent = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """
    Mocks out the underlying 'download_arxiv_paper' and 'query_results'
    tools so that no real HTTP calls are made during tests.
    """
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.paper_download."
            "download_arxiv_input.download_arxiv_paper"
        ) as mock_download_arxiv_paper,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.query_results.query_results"
        ) as mock_query_results,
    ):
        # Set return values to confirm agent is correctly orchestrating calls:
        mock_download_arxiv_paper.return_value = {
            "update": {
                "pdf_data": {"dummy_key": "dummy_value"},
                "messages": [],
            }
        }
        mock_query_results.return_value = {
            "update": {"results": "Mocked Query Results", "messages": []}
        }

        yield [mock_download_arxiv_paper, mock_query_results]


def test_paper_download_agent_initialization():
    """
    Ensures the paper download agent initializes properly using a mocked Hydra config.
    """
    thread_id = "test_thread_paper_dl"
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create_agent:
        mock_create_agent.return_value = mock.Mock()
        app = get_app(thread_id)
        assert app is not None, "The agent app should be successfully created."
        mock_create_agent.assert_called_once()


def test_paper_download_agent_invocation(mock_tools_fixture):
    """
    Verifies the agent can be invoked with a user query and provides
    updated state in return.
    """
    thread_id = "test_thread_paper_dl"
    initial_state = Talk2Scholars(messages=[HumanMessage(content="Download paper 1234.5678")])

    # Mock the model creation so we don't rely on actual LLM calls:
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create_agent:
        mock_agent = mock.Mock()
        # Simulate the agent returning some recognized AIMessage plus a synthetic update
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Here is the paper")],
            "pdf_data": {"file_bytes": b"FAKE_PDF_CONTENTS"},
        }
        mock_create_agent.return_value = mock_agent

        app = get_app(thread_id)
        result = app.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )

        assert "messages" in result, "The state must include the conversation messages."
        assert "pdf_data" in result, "Expected 'pdf_data' to be part of the updated state."
        mock_agent.invoke.assert_called_once()


def test_paper_download_agent_tools_assignment(mock_tools_fixture):
    """
    Checks that the correct tools (download_arxiv_paper, query_results)
    have been assigned to the agent node.
    """
    thread_id = "test_thread_paper_dl"
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
        mock_tool_instance.tools = mock_tools_fixture
        mock_toolnode.side_effect = lambda tool: mock_tool_instance

        app = get_app(thread_id)
        assert app is not None
        # We anticipate exactly two tools: download_arxiv_paper + query_results
        assert len(mock_tools_fixture) == 2, "Paper download agent must have exactly 2 tools."
        mock_toolnode.assert_called()


def test_paper_download_agent_hydra_failure():
    """
    Confirms the agent gracefully handles exceptions if Hydra config fails.
    """
    thread_id = "test_thread_paper_dl"
    with mock.patch("hydra.initialize", side_effect=Exception("Mock Hydra failure")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)
        assert "Mock Hydra failure" in str(exc_info.value), (
            "Expected the code to raise an exception when Hydra cannot initialize."
        )
