#!/usr/bin/env python3

"""
This tool is used to display the table of studies.
"""

import logging
from typing import Annotated, Dict, Any
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


@tool("display_results")
def display_results(state: Annotated[dict, InjectedState]) -> Dict[str, Any]:
    """
    Display the papers in the state. If no papers are found, raises an exception
    indicating that a search is needed.

    Args:
        state (dict): The state of the agent containing the papers.

    Returns:
        dict: A dictionary containing the papers and multi_papers from the state.

    Raises:
        NoPapersFoundError: If no papers are found in the state.

    Note:
        The exception allows the LLM to make a more informed decision about initiating a search.
    """
    logger.info("Displaying papers from the state")

    if not state.get("papers") and not state.get("multi_papers"):
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )

    return {
        "papers": state.get("papers"),
        "multi_papers": state.get("multi_papers"),
    }
