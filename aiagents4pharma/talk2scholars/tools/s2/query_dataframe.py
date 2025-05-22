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
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


class QueryDataFrameInput(BaseModel):
    """
    Pydantic schema for querying the metadata of displayed papers.

    Fields:
      question: A free-text prompt or Python expression to query the papers DataFrame.
      tool_call_id: LangGraph-injected identifier for tracking the tool invocation.
      state: Agent state dictionary. Must include:
        - 'last_displayed_papers': dictionary of paper metadata (rows = papers).
        - 'llm_model': model used to instantiate the DataFrame agent.

    Notes:
      - This tool is only for metadata queries. It does not perform full-text PDF analysis.
      - You can access standard metadata columns such as 'title', 'authors', 'venue', 'year', and 'arxiv_id'.
      - The following columns are typically excluded from direct querying for safety or irrelevance:
        ['Abstract', 'Key', 'semantic_scholar_paper_id', 'source', 'filename', 'pdf_url', 'attachment_key'].
    """

    question: str = Field(
        description=(
            "The metadata query to run over the papers DataFrame. Can be natural language "
            "(e.g., 'List all titles by author X') or Python code "
            "(e.g., df['arxiv_id'].dropna().tolist())."
        )
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool("query_dataframe", args_schema=QueryDataFrameInput, parse_docstring=True)
def query_dataframe(
    question: str, state: Annotated[dict, InjectedState], tool_call_id: str
) -> Command:
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
        tool_call_id (str): LangGraph-injected identifier for this tool call.

    Returns:
        Command: A structured response containing a ToolMessage with the query result.

    Raises:
        NoPapersFoundError: If no papers have been displayed yet.
    """
    logger.info("Querying last displayed papers with question: %s", question)
    llm_model = state.get("llm_model")
    if llm_model is None:
        raise ValueError("Missing 'llm_model' in state.")

    context_val = state.get("last_displayed_papers")
    if not context_val:
        logger.info("No papers displayed so far, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )

    # Resolve the paper dictionary
    if isinstance(context_val, dict):
        dic_papers = context_val
    else:
        dic_papers = state.get(context_val)

    if not isinstance(dic_papers, dict):
        raise ValueError(
            "Could not resolve a valid metadata dictionary from 'last_displayed_papers'"
        )

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

    llm_result = df_agent.invoke({"input": question}, stream_mode=None)
    response_text = llm_result["output"]

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response_text,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
