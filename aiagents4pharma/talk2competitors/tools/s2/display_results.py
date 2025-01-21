#!/usr/bin/env python3

'''
This tool is used to display the table of studies.
'''

from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

@tool('display_results')
def display_results(state: Annotated[dict, InjectedState]):
    """
    Display the table of studies.

    Args:
        state (dict): The state of the agent.
    """
    print ('Called display_results')
    return state["papers"]
