#!/usr/bin/env python3

"""
This tool is used to display the table of studies.
"""

import logging
from typing import Annotated
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


@tool("query_results", parse_docstring=True)
def query_results(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Query the last displayed papers from the state. If no papers are found,
    raises an exception.

    Use this also to get the last displayed papers from the state,
    and then use the papers to get recommendations for a single paper or
    multiple papers.

    Args:
        question (str): The question to ask the agent.
        state (dict): The state of the agent containing the papers.

    Returns:
        str: A message with the last displayed papers.
    """
    logger.info("Querying last displayed papers with question: %s", question)
    llm_model = state.get("llm_model")
    if not state.get("last_displayed_papers"):
        logger.info("No papers displayed so far, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )
    context_key = state.get("last_displayed_papers","pdf_data")
    dic_papers = state.get(context_key)
    df_papers = pd.DataFrame.from_dict(dic_papers, orient="index")
    df_agent = create_pandas_dataframe_agent(
        llm_model,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        df=df_papers,
        max_iterations=5,
        include_df_in_prompt=True,
        number_of_head_rows=df_papers.shape[0],
        verbose=True,
    )
    llm_result = df_agent.invoke(question, stream_mode=None)
    return llm_result["output"]
