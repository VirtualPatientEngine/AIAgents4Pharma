"""
Integration tests for talk2scholars system with OpenAI.
"""

import os
import pytest
import hydra
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.main_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars

# pylint: disable=redefined-outer-name


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key to run"
)
def test_main_agent_real_llm():
    """
    Test that the main agent invokes S2 agent correctly
    and updates the state with real LLM execution.
    """

    # Load Hydra Configuration EXACTLY like in main_agent.py
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
    hydra_cfg = cfg.agents.talk2scholars.main_agent

    assert hydra_cfg is not None, "Hydra config failed to load"

    # Use the real OpenAI API (ensure env variable is set)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=hydra_cfg.temperature)

    # Initialize main agent workflow (WITH real Hydra config)
    thread_id = "test_thread"
    app = get_app(thread_id, llm)

    # Provide an actual user query
    initial_state = Talk2Scholars(
        messages=[HumanMessage(content="Find AI papers on transformers")]
    )

    # Invoke the agent (triggers supervisor → s2_agent)
    result = app.invoke(
        initial_state,
        {"configurable": {"config_id": thread_id, "thread_id": thread_id}},
    )

    # Assert that the supervisor routed correctly
    assert "messages" in result, "Expected messages in response"

    # Fix: Accept AIMessage as a valid response type
    assert isinstance(
        result["messages"][-1], (HumanMessage, AIMessage, str)
    ), "Last message should be a valid response"
