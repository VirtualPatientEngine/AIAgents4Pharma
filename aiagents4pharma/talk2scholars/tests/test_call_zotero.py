"""
Integration tests for calling zotero_agent through the main_agent
"""

from unittest.mock import MagicMock
import pytest
from langgraph.types import Command
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from aiagents4pharma.talk2scholars.agents.main_agent import get_app
from aiagents4pharma.talk2scholars.state.state_talk2scholars import Talk2Scholars

# pylint: disable=redefined-outer-name
LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

@pytest.fixture
def test_state():
    """Creates an initial state for integration testing."""
    return Talk2Scholars(messages=[HumanMessage(content="Retrieve my Zotero papers.")])


@pytest.fixture
def mock_zotero_agent():
    """Mock the Zotero agent to return a predefined response."""
    mock_app = MagicMock()
    mock_app.invoke.return_value = {
        "messages": [
            HumanMessage(content="Retrieve my Zotero papers."),
            AIMessage(
                content="Here are your saved Zotero papers."
            ),  # Ensure this is returned
        ],
        "zotero_read": {"paper1": "A Zotero saved paper"},  # Ensure state is updated
        "last_displayed_papers": {},
    }
    return mock_app


@pytest.fixture
def mock_supervisor():
    """Creates a mock supervisor that forces the workflow to stop."""

    def mock_supervisor_node(state):
        """Force the workflow to terminate after calling zotero_agent."""
        # Ensure the response from Zotero agent is present in the state before ending
        if "messages" in state and len(state["messages"]) > 1:
            return Command(goto=END)  # End only after ensuring the state update
        return Command(goto="zotero_agent")  # Retry if state is not updated

    return mock_supervisor_node


def test_zotero_integration(
    test_state, mock_zotero_agent, mock_supervisor, monkeypatch
):
    """Runs the full LangGraph workflow to test `call_zotero_agent` execution."""

    # Patch `zotero_agent.get_app` to return the mock agent
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.zotero_agent.get_app",
        lambda *args, **kwargs: mock_zotero_agent,
    )

    # Patch `make_supervisor_node` to force termination after `zotero_agent`
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.make_supervisor_node",
        lambda *args, **kwargs: mock_supervisor,
    )

    # Initialize the LangGraph application
    app = get_app(thread_id="test_thread", llm_model=LLM_MODEL)

    # Run the full workflow (mocked Zotero agent is called)
    result = app.invoke(
        test_state,
        {
            "configurable": {
                "thread_id": "test_thread",
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint",
            }
        },
    )

    # Extract message content for assertion
    result_messages = [msg.content for msg in result["messages"]]

    # Assertions: Verify correct state updates
    assert "Retrieve my Zotero papers." in result_messages  # User query
    assert (
        "Here are your saved Zotero papers." in result_messages
    )  # AI response is present
    assert result["zotero_read"] == {"paper1": "A Zotero saved paper"}  # Data exists
