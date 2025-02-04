#!/usr/bin/env python3

'''
This is the state file for the Talk2BioModels agent.
'''

from typing import Annotated
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState

def add_data(data1: dict, data2: dict) -> dict:
    """
    Add two dictionaries.
    """
    if data1 is None and data2 is None:
        return []
    elif data1 is None:
        return data2
    elif data2 is None:
        return data1
    else:
        print (len(data1))
        print (len(data2))
        left_idx_by_name = {data['name']: idx for idx, data in enumerate(data1)}
        merged = data1.copy()
        # idx_to_remove = set()
        for data in data2:
            idx = left_idx_by_name.get(data['name'])
            if idx is not None:
                merged[idx] = data
                # idx_to_remove.add(idx)
            else:
                merged.append(data)
        print ("----------------")
        # return data1 + data2
        return merged
    # if len(data2) > 0:
    #     type(data2)
    # return {**data1, **data2}

class Talk2Biomodels(AgentState):
    """
    The state for the Talk2BioModels agent.
    """
    llm_model: str
    # A StateGraph may receive a concurrent updates
    # which is not supported by the StateGraph.
    # Therefore, we need to use Annotated to specify
    # the operator for the sbml_file_path field.
    # https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE/
    model_id: Annotated[list, operator.add]
    sbml_file_path: Annotated[list, operator.add]
    dic_simulated_data: Annotated[list[dict], operator.add]
    dic_scanned_data: Annotated[list[dict], operator.add]
    dic_steady_state_data: Annotated[list[dict], add_data]
