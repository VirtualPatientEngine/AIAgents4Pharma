#!/usr/bin/env python3

'''
This is the state file for the Talk2BioModels agent.
'''

from typing import Annotated
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState

class Talk2Biomodels(AgentState):
    """
    The state for the Talk2BioModels agent.
    """
    model_id: Annotated[list, operator.add]
    # sbml_file_path: str
    # A StateGraph may receive a concurrent updates
    # which is not supported by the StateGraph.
    # Therefore, we need to use Annotated to specify
    # the operator for the sbml_file_path field.
    # https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE/
    sbml_file_path: Annotated[list, operator.add]
    dic_simulated_data: dict
    llm_model: str