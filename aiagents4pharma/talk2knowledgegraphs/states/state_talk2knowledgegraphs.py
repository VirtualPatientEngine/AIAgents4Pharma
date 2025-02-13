"""
This is the state file for the Talk2KnowledgeGraphs agent.
"""

from typing import Annotated
# import operator
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import AgentState


def add_data(data1: dict, data2: dict) -> dict:
    """
    A reducer function to merge two dictionaries.
    """
    left_idx_by_name = {data["name"]: idx for idx, data in enumerate(data1)}
    merged = data1.copy()
    for data in data2:
        idx = left_idx_by_name.get(data["name"])
        if idx is not None:
            merged[idx] = data
        else:
            merged.append(data)
    return merged


class Talk2KnowledgeGraphs(AgentState):
    """
    The state for the Talk2KnowledgeGraphs agent.
    """

    llm_model: BaseChatModel
    embedding_model: Embeddings
    uploaded_files: list
    topk_nodes: int
    topk_edges: int
    dic_source_graph: Annotated[list[dict], add_data]
    dic_extracted_graph: Annotated[list[dict], add_data]
