#!/usr/bin/env python3

'''
This tool is used to display the table of studies.
'''

import logging
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool('display_results')
def display_results(state: Annotated[dict, InjectedState]):
    """
    Display the papers in the state.

    Args:
        state (dict): The state of the agent.
    """
    logger.info("Displaying papers from the state")
    return state["papers"]
