#!/usr/bin/env python3

'''
This is the state file for the Talk2Cells agent.
'''

from langgraph.prebuilt.chat_agent_executor import AgentState

class Talk2Cells(AgentState):
    """
    The state for the Talk2Cells agent.
    """
    search_table: str
