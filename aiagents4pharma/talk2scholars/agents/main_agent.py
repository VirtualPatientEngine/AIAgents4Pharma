#!/usr/bin/env python3

"""
Main agent module for initializing and running the Talk2Scholars application.

This module sets up the hierarchical agent system using LangGraph and integrates
various sub-agents for handling different tasks such as semantic scholar, zotero,
PDF processing, and paper downloading.

Functions:
- get_app: Initializes and returns the LangGraph-based hierarchical agent system.
"""

import logging
import hydra
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from ..agents.s2_agent import get_app as get_app_s2
from ..agents.zotero_agent import get_app as get_app_zotero
from ..agents.pdf_agent import get_app as get_app_pdf
from ..agents.paper_download_agent import get_app as get_app_paper_download
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
    if hasattr(llm_model, "model_name"):
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
            get_app_paper_download(uniq_id, llm_model),  # pdf
            get_app_pdf(uniq_id, llm_model),  # paper download
        ],
        model=llm_model,
        state_schema=Talk2Scholars,
        # Full history is needed to extract
        # the tool artifacts
        output_mode="full_history",
        # Allow supervisor to resume control and chain multiple sub-agent calls
        add_handoff_back_messages=True,
        prompt=cfg.system_prompt,
    )

    # Compile and run
    app = workflow.compile(checkpointer=MemorySaver(), name="Talk2Scholars_MainAgent")

    return app
