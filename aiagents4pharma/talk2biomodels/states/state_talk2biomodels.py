#!/usr/bin/env python3

'''
This is the state file for the Talk2BioModels agent.
'''

from typing import Annotated
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

def add_data(data1: dict, data2: dict) -> dict:
    """
    A reducer function to merge two dictionaries.
    """
    left_idx_by_name = {data['name']: idx for idx, data in enumerate(data1)}
    merged = data1.copy()
    for data in data2:
        idx = left_idx_by_name.get(data['name'])
        if idx is not None:
            merged[idx] = data
        else:
            merged.append(data)
    return merged

class Talk2Biomodels(AgentState):
    """
    The state for the Talk2BioModels agent.
    """
    llm_model: BaseChatModel
    text_embedding_model: Embeddings
    pdf_file_name: str
    # A StateGraph may receive a concurrent updates
    # which is not supported by the StateGraph. Hence,
    # we need to add a reducer function to handle the
    # concurrent updates.
    # https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE/
    model_id: Annotated[list, operator.add]
    sbml_file_path: Annotated[list, operator.add]
    dic_simulated_data: Annotated[list[dict], add_data]
    dic_scanned_data: Annotated[list[dict], add_data]
    dic_steady_state_data: Annotated[list[dict], add_data]
    dic_annotations_data : Annotated[list[dict], add_data]
