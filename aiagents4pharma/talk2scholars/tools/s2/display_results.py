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


@tool("display_results")
def display_results(state: Annotated[dict, InjectedState]) -> Dict[str, Any]:
    """
    Display the papers in the state. If no papers are found, indicates that a search is needed.

    Args:
        state (dict): The state of the agent containing the papers.

    Returns:
        dict: A dictionary containing the papers and multi_papers from the state, or a message
              indicating that a search needs to be performed if no papers are found.

    Note:
        Updates state directly to indicate search requirement if papers are not found.
    """
    logger.info("Displaying papers from the state")

    if not state.get("papers") and not state.get("multi_papers"):
        logger.info("No papers found in state, indicating search is needed")
        return "No papers found. A search needs to be performed first."

    return {
        "papers": state.get("papers"),
        "multi_papers": state.get("multi_papers"),
    }
