"""
Routing logic for zotero_agent through the main_agent
"""

import pytest
from langgraph.types import Command
from langgraph.graph import END
from langchain_core.messages import HumanMessage
from aiagents4pharma.talk2scholars.state.state_talk2scholars import Talk2Scholars

# pylint: disable=redefined-outer-name


@pytest.fixture
def mock_state():
    """Creates a mock state to simulate an ongoing conversation."""
    return Talk2Scholars(messages=[])


@pytest.fixture
def mock_router():
    """Creates a mock supervisor router that routes based on keyword matching."""

    def mock_supervisor_node(state):
        """Mock supervisor node that routes based on keyword matching."""
        query = state["messages"][-1].content.lower()
        # Define keywords for each sub-agent.
        s2_keywords = [
            "paper",
            "research",
            "citations",
            "journal",
            "articles",
            "references",
        ]
        zotero_keywords = ["zotero", "library", "saved papers", "academic library"]
        pdf_keywords = ["pdf", "document", "read pdf"]
        paper_download_keywords = ["download", "arxiv", "fetch paper", "paper download"]

        # Priority ordering: Zotero, then paper download, then PDF, then S2.
        if any(keyword in query for keyword in zotero_keywords):
            return Command(goto="zotero_agent")
        if any(keyword in query for keyword in paper_download_keywords):
            return Command(goto="paper_download_agent")
        if any(keyword in query for keyword in pdf_keywords):
            return Command(goto="pdf_agent")
        if any(keyword in query for keyword in s2_keywords):
            return Command(goto="s2_agent")
        # Default to end if no keyword matches.
        return Command(goto=END)

    return mock_supervisor_node


@pytest.mark.parametrize(
    "user_query,expected_agent",
    [
        ("Find papers on deep learning.", "s2_agent"),
        ("Show me my saved references in Zotero.", "zotero_agent"),
        ("I need some research articles.", "s2_agent"),
        ("Fetch my academic library.", "zotero_agent"),
        ("Retrieve citations.", "s2_agent"),
        ("Can you get journal articles?", "s2_agent"),
        ("I want to read the PDF document.", "pdf_agent"),
        ("Download the paper from arxiv.", "paper_download_agent"),
        ("Completely unrelated query.", "__end__"),
    ],
)
def test_routing_logic(mock_state, mock_router, user_query, expected_agent):
    """Tests that the routing logic correctly assigns the right agent or ends conversation."""
    mock_state["messages"].append(HumanMessage(content=user_query))
    result = mock_router(mock_state)

    assert result.goto == expected_agent, f"Failed for query: {user_query}"
