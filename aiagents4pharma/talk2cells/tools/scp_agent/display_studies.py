#!/usr/bin/env python3

'''
This tool is used to display the table of studies.
'''

import logging
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool('display_studies')
def display_studies(state: Annotated[dict, InjectedState]):
    """
    Display the table of studies.

    Args:
        state (dict): The state of the agent.
    """
    logger.log(logging.INFO, "Calling the tool display_studies")
    return state["search_table"]
