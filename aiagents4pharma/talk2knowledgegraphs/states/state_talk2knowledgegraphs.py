'''
This is the state file for the Talk2KnowledgeGraphs agent.
'''

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import AgentState


class Talk2KnowledgeGraphs(AgentState):
    """
    The state for the Talk2KnowledgeGraphs agent.
    """
    llm_model: BaseChatModel
    embedding_model: Embeddings
    uploaded_files: list
    topk_nodes: int
    topk_edges: int
    graph_dict: dict
    graph_text: str
    graph_summary: str
    input_tkg: str
    input_text_tkg: str
