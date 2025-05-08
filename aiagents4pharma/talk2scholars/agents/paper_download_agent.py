#!/usr/bin/env python3
"""
This module defines the paper download agent that connects to the arXiv API to fetch
paper details and PDFs. It is part of the Talk2Scholars project.
"""

import logging
from typing import Any, Dict
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.paper_download.download_arxiv_input import download_arxiv_paper

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_app(uniq_id, llm_model: BaseChatModel):
    """
    Initializes and returns the LangGraph application for the Talk2Scholars paper download agent.

    Args:
        uniq_id (str): A unique identifier for tracking the current session.
        llm_model (BaseChatModel, optional): The language model to be used by the agent.
            Defaults to ChatOpenAI(model="gpt-4o-mini", temperature=0.5).

    Returns:
        StateGraph: A compiled LangGraph application that enables the paper download agent to
            process user queries and retrieve arXiv papers.
    """

    # Load Hydra configuration
    logger.info("Loading Hydra configuration for Talk2Scholars paper download agent")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["agents/talk2scholars/paper_download_agent=default"],
        )
        cfg = cfg.agents.talk2scholars.paper_download_agent

    # Define tools properly
    tools = ToolNode([download_arxiv_paper])

    # Define the model
    logger.info("Using OpenAI model %s", llm_model)
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        prompt=cfg.paper_download_agent,
        checkpointer=MemorySaver(),
    )

    def paper_download_agent_node(state: Talk2Scholars) -> Dict[str, Any]:
        """
        Processes the current state to fetch the arXiv paper.
        """
        logger.info("Creating paper download agent node with thread_id: %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return result

    # Define new graph
    workflow = StateGraph(Talk2Scholars)

    # Adding node for paper download agent
    workflow.add_node("paper_download_agent", paper_download_agent_node)

    # Entering into the agent
    workflow.add_edge(START, "paper_download_agent")

    # Memory management for states between graph runs
    checkpointer = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer, name="paper_download_agent")

    # Logging the information and returning the app
    logger.info("Compiled the graph")
    return app
