#!/usr/bin/env python3
"""
This module defines the arXiv agent that connects to the arXiv API to fetch paper details and PDFs.
It is part of the Talk2Scholars project.
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
from ..tools.arxiv.download_pdf_arxiv import fetch_arxiv_paper
from ..tools.s2.query_results import query_results as arxiv_query_results
#initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model: BaseChatModel = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)):
    """
    Create and return the langgraph app for the Talk2Scholars arXiv agent.

    This function sets up the workflow, tools, and language model needed to fetch arXiv papers.

    Parameters:
        uniq_id: A unique identifier for tracking this session.
        llm_model: The language model to use (default is ChatOpenAI with the "gpt-4o-mini" model).

    Returns:
        The compiled langgraph app ready to run.
    """

    def agent_arxiv_node(state: Talk2Scholars) -> Dict[str,Any]:
        """

        Get the arXiv paper using the paper ID from the state.

        This function takes the current state, which contains the paper ID,
        calls the model to fetch the paper, and returns the result.

        Parameters:
            state: The current state with the paper ID and related data.

        Returns:
            A dictionary with the result of the fetch operation.

        """
        logger.log(logging.INFO, "Creating Agent Arxiv node with thread_id: %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    logger.log(logging.INFO, "thread_id, llm_model: %s, %s", uniq_id, llm_model)

    #load the configuration of hydras
    logger.log(logging.INFO, "Loading Hydra configuration for talk2scholars arxiv agent")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/arxiv_agent=default"]
            )
        cfg = cfg.agents.talk2scholars.arxiv_agent

#define the tools
    tools = ToolNode([fetch_arxiv_paper, arxiv_query_results])

#define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)

#create the agent
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=cfg.arxiv_agent,
        checkpointer=MemorySaver(),
    )
#define new graph
    workflow = StateGraph(Talk2Scholars)

#defining 2 cycle nodes
    workflow.add_node("agent_arxiv", agent_arxiv_node)

#entering into the agent
    workflow.add_edge(START, "agent_arxiv")

#starting memory of states between graph runs
    checkpointer = MemorySaver()

#compiling the graph
    app = workflow.compile(checkpointer=checkpointer)

#logging the information and returning the app
    logger.log(logging.INFO, "Compiled the graph")
    return app
