#!/usr/bin/env python3

"""
Query DataFrame Tool

This LangGraph tool answers metadata-level questions over the last displayed set of papers.
It loads the papers (titles, authors, dates, URLs, etc.) into a pandas DataFrame and
invokes an LLM-powered DataFrame agent to process queries such as:
    - "Which papers were published after 2020?"
    - "List authors of paper X"
    - "Filter papers with 'Transformer' in the title"

Note: This tool is only for tabular metadata queries. For PDF content Q&A,
use the `question_and_answer` tool.
"""

import logging
from typing import Annotated
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


class QueryDataFrameInput(BaseModel):
    """
    Input schema for the Query DataFrame tool.

    Attributes:
        question (str): Metadata-level query to run over the last displayed papers.
        state (dict): Shared agent state containing 'last_displayed_papers' and metadata.
    """
    question: str = Field(
        ..., description="The metadata-level query to execute on the papers table"
    )
    state: Annotated[dict, InjectedState] = Field(
        ..., description="Injected shared state with 'last_displayed_papers' key"
    )

@tool(args_schema=QueryDataFrameInput, parse_docstring=True)
def query_dataframe(
    question: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Answer a metadata query over the last displayed papers using a DataFrame agent.

    This tool retrieves the key 'last_displayed_papers' from the shared state, loads
    the corresponding article metadata into a pandas DataFrame, and uses an LLM-driven
    pandas agent to execute the user's query on that table.

    Args:
        question (str): A metadata-level question (e.g. filters, selections) to run
                        against the papers table.
        state (dict): Injected agent state containing:
            - 'last_displayed_papers': key name under which the metadata dict is stored
            - The metadata dict itself at state[last_displayed_papers]
            - 'llm_model': the LLM instance to power the DataFrame agent

    Returns:
        str: The agent's natural-language answer to the metadata query.

    Raises:
        NoPapersFoundError: If no papers have been loaded into state for querying.
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
