"""
Integration tests for calling zotero_agent through the main_agent
"""

import logging
import pytest
from langchain_core.messages import HumanMessage
from aiagents4pharma.talk2scholars.agents.main_agent import get_app
from aiagents4pharma.talk2scholars.state.state_talk2scholars import Talk2Scholars

# pylint: disable=redefined-outer-name


@pytest.fixture
def test_state():
    """Creates an initial state for integration testing."""
    return Talk2Scholars(messages=[HumanMessage(content="Retrieve my Zotero papers.")])


def test_zotero_integration(test_state, caplog):
    """Runs the full LangGraph workflow to test `call_zotero_agent` execution."""

    # Capture logs to verify that `call_zotero_agent` is actually executed
    with caplog.at_level(logging.INFO):

        # Initialize the LangGraph application
        app = get_app(thread_id="test_thread")

        # Run the full workflow (real Zotero agent is called)
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

    # Debugging Output
    print("\nDEBUG: Full Workflow Result Messages:", result_messages)

    # Assertions
    assert "Retrieve my Zotero papers." in result_messages  # User query
    assert any("Zotero" in msg for msg in result_messages)  # Zotero response expected
    assert "zotero_read" in result and isinstance(
        result["zotero_read"], dict
    )  # Data exists

    # Ensure logs confirm `call_zotero_agent` was invoked
    assert "Calling Zotero agent" in caplog.text  # Log entry before calling
    assert "Zotero agent completed with response" in caplog.text  # Log after completion

    print("\nTest Passed: Full integration of `call_zotero_agent` confirmed!")
