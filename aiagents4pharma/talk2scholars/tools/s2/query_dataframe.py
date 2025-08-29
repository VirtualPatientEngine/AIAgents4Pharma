#!/usr/bin/env python3


"""
Query the metadata table of the most recently displayed papers.

This tool loads `state['last_displayed_papers']` into a pandas DataFrame and uses an
LLM-driven DataFrame agent to execute metadata-level queries. It supports both
natural-language prompts (e.g., “list titles by author X”) and direct Python expressions
over the DataFrame.

Capabilities
- Filter, sort, and aggregate rows using metadata columns (e.g., Title, Authors, Venue, Year).
- Extract paper identifiers from a designated column (default: 'paper_ids'),
  optionally for a single row.
- Return the DataFrame agent’s textual result as a ToolMessage.

Requirements
- `state['llm_model']`: model used to instantiate the DataFrame agent.
- `state['last_displayed_papers']`: dictionary mapping row keys → metadata records.

Notes
- Operates strictly on the metadata table; it does not parse or read PDF content.
- When `extract_ids=True`, the tool constructs a Python expression for the agent to evaluate
  and return identifiers from `id_column`. If `row_number` is provided (1-based), only that row’s
  first identifier is returned; otherwise a list is returned from all rows that have values.
"""

import logging
from typing import Annotated, Any

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
    Input schema for querying the last displayed papers metadata DataFrame.

    Fields:
      question (str):
        The query to execute. Accepts natural language (e.g., "List titles from 2024")
        or a Python expression over the DataFrame (e.g., "df['Title'].tolist()").

      extract_ids (bool, default=False):
        When True, the tool prepares a Python expression for the DataFrame agent to extract
        identifiers from `id_column`. Use to obtain IDs from the metadata table.

      id_column (str, default="paper_ids"):
        Name of the column that contains per-row lists of identifiers (e.g., ["arxiv:2301.12345"]).
        Used only when `extract_ids=True`.

      row_number (int | None, default=None):
        1-based row index. When provided with `extract_ids=True`, returns only that row’s first
        identifier. When omitted, returns a list of first identifiers from each applicable row.

      tool_call_id (InjectedToolCallId):
        Internal identifier for tracing the tool invocation.

      state (dict):
        Agent state containing:
          - 'last_displayed_papers': dict with the current results table (rows → metadata)
          - 'llm_model': model object or reference for the DataFrame agent
    """

    question: str = Field(
        description=(
            "The metadata query to run over the papers DataFrame. Can be natural language "
            "(e.g., 'List all titles by author X') or Python code "
            "(e.g., df['arxiv_id'].dropna().tolist())."
        )
    )
    extract_ids: bool = Field(
        default=False,
        description=(
            "If true, instruct the DataFrame agent to extract values from the"
            "specified ID column via a Python expression."
        ),
    )
    id_column: str = Field(
        default="paper_ids",
        description=(
            "Name of the metadata column containing a list of paper IDs to"
            "extract when extract_ids=True."
        ),
    )
    row_number: int | None = Field(
        default=None,
        description=(
            "1-based index of the ID to extract from the list; if provided, returns only"
            "that single ID."
        ),
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool(
    "query_dataframe",
    args_schema=QueryDataFrameInput,
    parse_docstring=True,
)
def query_dataframe(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: str,
    **kwargs: Any,
) -> Command:
    """
    Execute a metadata query against the DataFrame built from `last_displayed_papers`.

    Behavior
    - Builds a pandas DataFrame from `state['last_displayed_papers']`.
    - Instantiates a pandas DataFrame agent with `state['llm_model']`.
    - Runs either:
        • the provided natural-language prompt, or
        • a constructed Python expression when `extract_ids=True`
          (optionally scoped to `row_number`, 1-based).
    - Returns the DataFrame agent’s output text in a ToolMessage.

    Parameters
      question (str):
        Natural-language query or Python expression to run on the DataFrame.
      state (dict):
        Must provide 'llm_model' and 'last_displayed_papers'.
      tool_call_id (str):
        Internal identifier for the tool call.
      **kwargs:
        extract_ids (bool): Enable ID extraction from `id_column`.
        id_column (str): Column containing lists of identifiers (default: "paper_ids").
        row_number (int | None): 1-based index for a single-row extraction.

    Returns
      Command:
        update = {
          "messages": [
            ToolMessage(
              content=<text result from the DataFrame agent>,
              tool_call_id=<tool_call_id>
            )
          ]
        }

    Errors
    - Raises `ValueError` if 'llm_model' is missing in `state`.
    - Raises `NoPapersFoundError` if `state['last_displayed_papers']` is missing or empty.
    - Raises `ValueError` if a required argument for the chosen mode is invalid
      (e.g., no `id_column` when `extract_ids=True`).

    Examples
    - Natural language:
        question="List titles where Year >= 2023"
    - Python list of titles:
        question="df.query('Year >= 2023')['Title'].tolist()"
    - Extract first ID from row 1:
        extract_ids=True, row_number=1
    - Extract first IDs from all rows:
        extract_ids=True
    """
    logger.info("Querying last displayed papers with question: %s", question)
    llm_model = state.get("llm_model")
    if llm_model is None:
        raise ValueError("Missing 'llm_model' in state.")

    context_val = state.get("last_displayed_papers")
    if not context_val:
        logger.info("No papers displayed so far, raising NoPapersFoundError")
        raise NoPapersFoundError("No papers found. A search needs to be performed first.")

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
    # Prepare the query: if extracting IDs, let the DataFrame agent handle it via Python code
    extract_ids_flag = kwargs.get("extract_ids", False)
    id_column = kwargs.get("id_column", "paper_ids")
    row_number = kwargs.get("row_number")
    question_to_agent = question
    if extract_ids_flag:
        if not id_column:
            raise ValueError("Must specify 'id_column' when extract_ids=True.")
        if row_number is not None:
            question_to_agent = f"df['{id_column}'].dropna().str[0].tolist()[{row_number - 1}]"
        else:
            question_to_agent = f"df['{id_column}'].dropna().str[0].tolist()"
        logger.info("extract_ids enabled: asking agent to run expression: %s", question_to_agent)

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

    llm_result = df_agent.invoke({"input": question_to_agent}, stream_mode=None)
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
