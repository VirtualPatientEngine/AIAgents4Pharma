#!/usr/bin/env python3

"""
Main agent for the talk2scholars app using ReAct pattern.

This module implements a hierarchical agent system where a supervisor agent
routes queries to specialized sub-agents. It follows the LangGraph patterns
for multi-agent systems and implements proper state management.
"""

import logging
import hydra
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from ..agents.s2_agent import get_app as get_app_s2
from ..agents.zotero_agent import get_app as get_app_zotero
from ..state.state_talk2scholars import Talk2Scholars

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_app(uniq_id, llm_model: BaseChatModel):
    """
    Initializes and returns the LangGraph-based hierarchical agent system.

    This function constructs the agent workflow by defining nodes for the supervisor
    and sub-agents. It compiles the graph using `StateGraph` to enable structured
    conversational workflows.

    Args:
        thread_id (str): A unique session identifier for tracking conversation state.
        llm_model (BaseChatModel, optional): The language model used for query processing.
            Defaults to `ChatOpenAI(model="gpt-4o-mini", temperature=0)`.

    Returns:
        StateGraph: A compiled LangGraph application that can process user queries.

    Example:
        >>> app = get_app("thread_123")
        >>> result = app.invoke(initial_state)
    """
    if llm_model.model_name == "gpt-4o-mini":
        llm_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            model_kwargs={"parallel_tool_calls": False},
        )
    # Load hydra configuration
    logger.log(logging.INFO, "Launching Talk2Scholars with thread_id %s", uniq_id)
    with hydra.initialize(version_base=None, config_path="../configs/"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.main_agent
    logger.log(logging.INFO, "System_prompt of Talk2Scholars: %s", cfg.system_prompt)
    # Create supervisor workflow
    workflow = create_supervisor(
        [
            get_app_s2(uniq_id, llm_model),  # semantic scholar
            get_app_zotero(uniq_id, llm_model),  # zotero
        ],
        model=llm_model,
        state_schema=Talk2Scholars,
        # Full history is needed to extract
        # the tool artifacts
        output_mode="full_history",
        add_handoff_back_messages=False,
        prompt=cfg.system_prompt,
    )

    # Compile and run
    app = workflow.compile(checkpointer=MemorySaver(), name="Talk2Scholars_MainAgent")

    return app
