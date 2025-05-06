#!/usr/bin/env python3


"""
Tool for rendering the most recently displayed papers as a DataFrame artifact for the front-end.

This module defines a tool that retrieves the paper metadata stored under the state key
'last_displayed_papers' and returns it as an artifact (dictionary of papers). The front-end
can then render this artifact as a pandas DataFrame for display. If no papers are found,
a NoPapersFoundError is raised to indicate that a search or recommendation should be
performed first.
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


@tool("display_dataframe", parse_docstring=True)
def display_dataframe(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """
    Render the last set of retrieved papers as a DataFrame in the front-end.

    This function reads the 'last_displayed_papers' key from state, fetches the
    corresponding metadata dictionary, and returns a Command with a ToolMessage
    containing the artifact (dictionary) for the front-end to render as a DataFrame.
    If no papers are found in state, it raises a NoPapersFoundError to indicate
    that a search or recommendation must be performed first.

    Args:
        tool_call_id (InjectedToolCallId): Unique ID of this tool invocation.
        state (dict): The agent's state containing the 'last_displayed_papers' reference.

    Returns:
        Command: A command whose update contains a ToolMessage with the artifact
                 (papers dict) for DataFrame rendering in the UI.

    Raises:
        NoPapersFoundError: If no entries exist under 'last_displayed_papers' in state.
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
