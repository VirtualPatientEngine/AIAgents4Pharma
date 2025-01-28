'''
This is the state file for the Talk2KnowledgeGraphs agent.
'''

from typing import Annotated
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState

class Talk2KnowledgeGraphs(AgentState):
    """
    The state for the Talk2KnowledgeGraphs agent.
    """
    documents_path: Annotated[list, operator.add]
    llm_model: str
    graph_dict: dict
    graph_text: str
