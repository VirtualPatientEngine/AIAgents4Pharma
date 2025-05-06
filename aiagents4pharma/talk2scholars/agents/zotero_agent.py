#!/usr/bin/env python3

"""
Agent for interacting with Zotero with human-in-the-loop features
"""

import logging
from typing import Any, Dict
import hydra

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.zotero.zotero_read import zotero_read
from ..tools.zotero.zotero_review import zotero_review
from ..tools.zotero.zotero_write import zotero_write
from ..tools.s2.display_dataframe import display_dataframe
from ..tools.s2.query_dataframe import query_dataframe
from ..tools.s2.retrieve_semantic_scholar_paper_id import (
    retrieve_semantic_scholar_paper_id,
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_app(uniq_id, llm_model: BaseChatModel):
    """
    Initializes and returns the LangGraph application for the Zotero agent.

    This function sets up the Zotero agent, which integrates various tools to search,
    retrieve, and display research papers from Zotero. The agent follows the ReAct
    pattern for structured interaction and includes human-in-the-loop features.

    Args:
        uniq_id (str): Unique identifier for the current conversation session.
        llm_model (BaseChatModel, optional): The language model to be used by the agent.
            Defaults to `ChatOpenAI(model="gpt-4o-mini", temperature=0)`.

    Returns:
        StateGraph: A compiled LangGraph application that enables the Zotero agent to
            process user queries and retrieve research papers.

    Example:
        >>> app = get_app("thread_123")
        >>> result = app.invoke(initial_state)
    """

    def zotero_agent_node(state: Talk2Scholars) -> Dict[str, Any]:
        """
        Processes the user query and retrieves relevant research papers from Zotero.

        This function calls the language model using the configured `ReAct` agent to
        analyze the state and generate an appropriate response. The function then
        returns control to the main supervisor.

        Args:
            state (Talk2Scholars): The current conversation state, including messages exchanged
                and any previously retrieved research papers.

        Returns:
            Dict[str, Any]: A dictionary containing the updated conversation state.

        Example:
            >>> result = zotero_agent_node(current_state)
            >>> papers = result.get("papers", [])
        """
        logger.log(
            logging.INFO, "Creating Agent_Zotero node with thread_id %s", uniq_id
        )
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars Zotero agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["agents/talk2scholars/zotero_agent=default"],
        )
        cfg = cfg.agents.talk2scholars.zotero_agent
        logger.log(logging.INFO, "Loaded configuration for Zotero agent")

    # Define the tools
    tools = ToolNode(
        [
            zotero_read,
            display_dataframe,
            query_dataframe,
            retrieve_semantic_scholar_paper_id,
            zotero_review,
            zotero_write,
        ]
    )

    # Define the model
    logger.log(logging.INFO, "Using model %s", llm_model)

    # Create the agent
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        prompt=cfg.zotero_agent,
        checkpointer=MemorySaver(),  # Required for interrupts to work
    )

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("zotero_agent", zotero_agent_node)
    workflow.add_edge(START, "zotero_agent")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer, name="zotero_agent")
    logger.log(
        logging.INFO,
        "Compiled the graph with thread_id %s and llm_model %s",
        uniq_id,
        llm_model,
    )

    return app
