#!/usr/bin/env python3


"""
Tool for displaying search or recommendation results.

This module defines a tool that retrieves and displays a table of research papers
found during searches or recommendations. If no papers are found, an exception is raised
to signal the need for a new search.
"""


import logging

from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """
    Exception raised when no research papers are found in the agent's state.

    This exception helps the language model determine whether a new search
    or recommendation should be initiated.

    Example:
        >>> if not papers:
        >>>     raise NoPapersFoundError("No papers found. A search is needed.")
    """


@tool("display_results", parse_docstring=True)
def display_results(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """
    Displays retrieved research papers after a search or recommendation.

    This function retrieves the last displayed research papers from the state and
    returns them as an artifact for further processing. If no papers are found,
    it raises a `NoPapersFoundError` to indicate that a new search is needed.

    Args:
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID for tracking.
        state (dict): The agent's state containing retrieved papers.

    Returns:
        Command: A command containing a message with the number of displayed papers
                 and an attached artifact for further reference.

    Raises:
        NoPapersFoundError: If no research papers are found in the agent's state.

    Example:
        >>> state = {"last_displayed_papers": {"paper1": "Title 1", "paper2": "Title 2"}}
        >>> result = display_results(tool_call_id="123", state=state)
        >>> print(result.update["messages"][0].content)
        "2 papers found. Papers are attached as an artifact."
    """
    logger.info("Displaying papers")
    context_key = state.get("last_displayed_papers")
    artifact = state.get(context_key)
    if not artifact:
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search/rec needs to be performed first."
        )
    content = f"{len(artifact)} papers found. Papers are attached as an artifact."
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=artifact,
                )
            ],
        }
    )
