#!/usr/bin/env python3


"""
Tool for rendering the most recently displayed papers as a DataFrame artifact for the front-end.

Call this tool when you need to present the current set of retrieved papers to the user
(e.g., "show me the papers", "display results"). It reads the 'last_displayed_papers'
dictionary from the agent state and returns it as an artifact that the UI will render
as a pandas DataFrame. This tool does not perform any new searches or filtering; it
only displays the existing list. If no papers are available, it raises NoPapersFoundError
to signal that a search or recommendation must be executed first.
"""


import logging

from typing import Annotated
from pydantic import BaseModel, Field
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


class DisplayDataFrameInput(BaseModel):
    """
    Pydantic schema for displaying the last set of papers as a DataFrame artifact.

    Fields:
      state: Agent state dict containing the 'last_displayed_papers' key.
      tool_call_id: LangGraph-injected identifier for this tool invocation.
    """

    state: Annotated[dict, InjectedState] = Field(
        ..., description="Agent state containing the 'last_displayed_papers' reference."
    )
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        ..., description="LangGraph-injected identifier for this tool call."
    )


@tool(
    "display_dataframe",
    args_schema=DisplayDataFrameInput,
    parse_docstring=True,
)
def display_dataframe(
    tool_call_id: str,
    state: dict,
) -> Command:
    """
    Render the last set of retrieved papers as a DataFrame in the front-end.

    This function reads the 'last_displayed_papers' key from state, fetches the
    corresponding metadata dictionary, and returns a Command with a ToolMessage
    containing the artifact (dictionary) for the front-end to render as a DataFrame.
    If no papers are found in state, it raises a NoPapersFoundError to indicate
    that a search or recommendation must be performed first.

    Args:
        tool_call_id (str): LangGraph-injected unique ID for this tool call.
        state (dict): The agent's state containing the 'last_displayed_papers' reference.

    Returns:
        Command: A command whose update contains a ToolMessage with the artifact
                 (papers dict) for DataFrame rendering in the UI.

    Raises:
        NoPapersFoundError: If no entries exist under 'last_displayed_papers' in state.
    """
    logger.info("Displaying papers from 'last_displayed_papers'")
    context_val = state.get("last_displayed_papers")
    # Support both key reference (str) and direct mapping
    if isinstance(context_val, dict):
        artifact = context_val
    else:
        artifact = state.get(context_val)
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
