#!/usr/bin/env python3

"""
This tool is used to display the table of studies.
"""

import logging
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool("display_results")
def display_results(state: Annotated[dict, InjectedState]):
    """
    Display the papers in the state.

    Args:
        state (dict): The state of the agent containing the papers.

    Returns:
        dict: A dictionary containing the papers and multi_papers from the state.

    Raises:
        ValueError: If no papers are found in the state, trigger a search.
    """
    logger.info("Displaying papers from the state")

    if not state.get("papers") and not state.get("multi_papers"):
        logger.error("No papers found in the state. Triggering a search.")
        raise ValueError("No papers found in the state. Please trigger a search.")

    return {"papers": state["papers"], "multi_papers": state["multi_papers"]}
