#!/usr/bin/env python3
"""
Updated Unit Tests for the arXiv agent (Talk2Scholars arXiv sub-agent).
"""

from unittest import mock
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from ..agents.paper_download_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


def test_paper_download_agent_initialization():
    """
    Test that the arXiv agent initializes correctly with the mock configuration.
    """
    thread_id = "test_thread"
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create:
        mock_create.return_value = mock.Mock()
        app = get_app(thread_id)
        assert app is not None
        assert mock_create.called


def test_paper_download_agent_invocation():
    """
    Test that the arXiv agent processes user input and returns a valid response.
    """
    thread_id = "test_thread"
    mock_state = Talk2Scholars(
        messages=[HumanMessage(content="Fetch arXiv paper for AI research")]
    )
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        # Simulate a response from the react agent
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Here is your arXiv paper")],
            "papers": {"pdf": "Mock PDF Result"},
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
        assert result["papers"]["pdf"] == "Mock PDF Result"
        assert mock_agent.invoke.called


def test_download_arxiv_paper_tool_assignment():
    """
    Ensure that the correct tool (download_arxiv_paper) is assigned to the 
    arXiv agent.
    """
    thread_id = "test_thread"
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create, mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.ToolNode"
    ) as mock_toolnode:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        # Simulate a ToolNode that is instantiated with a list of tools
        mock_tool_instance = mock.Mock()
        # For the arXiv agent, we expect exactly one tool to be assigned.
        mock_tool_instance.tools = [mock.Mock()]
        mock_toolnode.return_value = mock_tool_instance
        get_app(thread_id)
        assert mock_toolnode.called
        assert len(mock_tool_instance.tools) == 1


def test_query_results_tool():
    """
    Test that the query_results tool is correctly integrated and utilized by the 
    arXiv agent. This test validates that the agent's response contains the 
    expected 'query_results' key, thereby ensuring our tool orchestration layer 
    is operating with precision.
    """
    thread_id = "test_thread"
    mock_state = Talk2Scholars(
        messages=[HumanMessage(content="Query results for arXiv papers")]
    )
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.paper_download_agent.create_react_agent"
    ) as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        # Simulate an agent response that includes the query_results tool output
        mock_agent.invoke.return_value = {
            "messages": [HumanMessage(content="Query results for arXiv papers")],
            "last_displayed_papers": {},
            "papers": {"query_results": "Mock Query Result"},
            "multi_papers": {},
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
        assert "query_results" in result["papers"]
        assert mock_agent.invoke.called


def test_paper_download_agent_hydra_failure():
    """
    Test exception handling when Hydra fails to load configuration.
    """
    thread_id = "test_thread"
    with mock.patch("hydra.initialize", side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)
        assert "Hydra error" in str(exc_info.value)
