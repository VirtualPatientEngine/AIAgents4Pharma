#!/usr/bin/env python3

"""
Tool for querying the metadata table of the last displayed papers.

This tool loads the most recently displayed papers into a pandas DataFrame and uses an
LLM-driven pandas agent to answer metadata-level questions (e.g., filter by author, list titles).
It is intended for metadata exploration only, and does not perform content-based retrieval
or summarization. For PDF-level question answering, use the 'question_and_answer_agent'.
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


@tool("query_dataframe", parse_docstring=True)
def query_dataframe(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Perform a tabular query on the most recently displayed papers.

    This function loads the last displayed papers into a pandas DataFrame and uses a
    pandas DataFrame agent to answer metadata-level questions (e.g., "Which papers have
    'Transformer' in the title?", "List authors of paper X"). It does not perform PDF
    content analysis or summarization; for content-level question answering, use the
    'question_and_answer_agent'.

    Args:
        question (str): The metadata query to ask over the papers table.
        state (dict): The agent's state containing 'last_displayed_papers'
            key referencing the metadata table in state.

    Returns:
        str: The LLM's response to the metadata query.

    Raises:
        NoPapersFoundError: If no papers have been displayed yet.
    """
    logger.info("Querying last displayed papers with question: %s", question)
    llm_model = state.get("llm_model")
    if not state.get("last_displayed_papers"):
        logger.info("No papers displayed so far, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )
    context_key = state.get("last_displayed_papers")
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
