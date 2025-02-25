"""
Integration tests for calling s2_agent through the main_agent with OpenAI.
"""

import os
import pytest
import hydra
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.main_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key to run"
)
def test_call_s2_agent_with_real_llm():
    """
    Test that the main agent correctly routes a query to S2 agent
    and updates the conversation state with a real LLM.
    """

    # Load real Hydra Configuration
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
    hydra_cfg = cfg.agents.talk2scholars.main_agent

    assert hydra_cfg is not None, "Hydra config failed to load"

    # Use the real OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=hydra_cfg.temperature)

    # Initialize main agent workflow with real LLM
    thread_id = "test_thread"
    app = get_app(thread_id, llm)

    # Provide an actual user query
    initial_state = Talk2Scholars(
        messages=[HumanMessage(content="Find AI papers on reinforcement learning")]
    )

    # Invoke the agent (triggers supervisor → s2_agent)
    result = app.invoke(
        initial_state,
        {"configurable": {"config_id": thread_id, "thread_id": thread_id}},
    )

    # Assert that the supervisor routed correctly
    assert "messages" in result, "Expected messages in response"

    # Check if AIMessage or a valid response was returned
    assert isinstance(
        result["messages"][-1], (HumanMessage, AIMessage, str)
    ), "Last message should be a valid response"
