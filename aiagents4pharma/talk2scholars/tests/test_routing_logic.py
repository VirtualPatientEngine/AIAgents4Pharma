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
        query = state["messages"][-1].content.lower()

        # Expanded keyword matching for S2 Agent
        s2_keywords = [
            "paper",
            "research",
            "citations",
            "journal",
            "articles",
            "references",
        ]
        zotero_keywords = ["zotero", "library", "saved papers", "academic library"]

        if any(keyword in query for keyword in zotero_keywords):
            return Command(goto="zotero_agent")
        if any(keyword in query for keyword in s2_keywords):
            return Command(goto="s2_agent")

        # If no match, default to ending the conversation
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
        (
            "Completely unrelated query.",
            "__end__",
        ),  # NEW: Should trigger the `END` case
    ],
)
def test_routing_logic(mock_state, mock_router, user_query, expected_agent):
    """Tests that the routing logic correctly assigns the right agent or ends conversation."""
    mock_state["messages"].append(HumanMessage(content=user_query))
    result = mock_router(mock_state)

    print(f"\nDEBUG: Query '{user_query}' routed to: {result.goto}")

    assert result.goto == expected_agent, f"Failed for query: {user_query}"
