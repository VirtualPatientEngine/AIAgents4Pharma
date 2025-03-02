#!/usr/bin/env python3
"""
This module defines the paper download agent that connects to the arXiv API to fetch
paper details and PDFs. It is part of the Talk2Scholars project.
"""

import logging
from typing import Any, Dict
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.paper_download.download_arxiv_paper import download_arxiv_paper
from ..tools.s2.query_results import query_results

#initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model: BaseChatModel = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)):
    """
    Initializes and returns the LangGraph application for the Talk2Scholars paper download agent.

    This function sets up the paper download agent, which integrates the necessary tools,
    language model, and workflow to fetch arXiv papers. The agent follows the ReAct pattern
    for structured interaction.

    Args:
        uniq_id (str): A unique identifier for tracking the current session.
        llm_model (BaseChatModel, optional): The language model to be used by the agent.
            Defaults to ChatOpenAI(model="gpt-4o-mini", temperature=0.5).

    Returns:
        StateGraph: A compiled LangGraph application that enables the paper download agent to
            process user queries and retrieve arXiv papers.

    Example:
        >>> app = get_app("thread_123")
        >>> result = app.invoke(initial_state)
    """


    def paper_download_agent_node(state: Talk2Scholars) -> Dict[str,Any]:
        """
        Processes the current state to fetch the arXiv paper.

        This function extracts the paper ID from the provided state, calls the language model
        via the ReAct agent to fetch the paper details and PDF, and then returns the updated state.

        Args:
            state (Talk2Scholars): The current state containing the paper ID and related data.

        Returns:
            Dict[str, Any]: A dictionary representing the updated state
            with the fetched paper details.
        """
        logger.log(logging.INFO, "Creating paper download agent node with thread_id: %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    logger.log(logging.INFO, "thread_id, llm_model: %s, %s", uniq_id, llm_model)

    #load the configuration of hydras
    logger.log(logging.INFO, "Loading Hydra configuration for talk2scholars paper download agent")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/paper_download_agent=default"]
            )
        cfg = cfg.agents.talk2scholars.paper_download_agent

#define the tools
    tools = ToolNode([download_arxiv_paper, query_results])

#define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)

#create the agent
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=cfg.paper_download_agent,
        checkpointer=MemorySaver(),
    )
    #define new graph
    workflow = StateGraph(Talk2Scholars)

    #defining 2 cycle nodes
    workflow.add_node("paper_download_agent", paper_download_agent_node)
    #place holder for pubmed tool

    #entering into the agent
    workflow.add_edge(START, "paper_download_agent")
    #place holder for pubmed tool

    #starting memory of states between graph runs
    checkpointer = MemorySaver()

    #compiling the graph
    app = workflow.compile(checkpointer=checkpointer)

    #logging the information and returning the app
    logger.log(logging.INFO, "Compiled the graph")
    return app
