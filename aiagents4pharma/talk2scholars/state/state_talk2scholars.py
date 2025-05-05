"""
State management for the Talk2Scholars agent.

This module defines the state class `Talk2Scholars`, which maintains the conversation
context, retrieved papers, and other relevant metadata. The state ensures consistency
across agent interactions.
"""

import logging
from typing import Annotated, Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_dict(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replaces the existing dictionary with a new dictionary.

    This function logs the state update and ensures that the old state is replaced
    with the new one.

    Args:
        existing (Dict[str, Any]): The current dictionary state.
        new (Dict[str, Any]): The new dictionary state to replace the existing one.

    Returns:
        Dict[str, Any]: The updated dictionary state.

    Example:
        >>> old_state = {"papers": {"id1": "Paper 1"}}
        >>> new_state = {"papers": {"id2": "Paper 2"}}
        >>> updated_state = replace_dict(old_state, new_state)
        >>> print(updated_state)
        {"papers": {"id2": "Paper 2"}}
    """
    # No-op operation to use the 'existing' variable
    _ = len(existing)
    return new


class Talk2Scholars(AgentState):
    """
    Represents the state of the Talk2Scholars agent.
    This class extends `AgentState` to maintain conversation history, retrieved papers,
    and interactions with the language model.
    Attributes:
        last_displayed_papers (Dict[str, Any]): Stores the most recently displayed papers.
        papers (Dict[str, Any]): Stores the research papers retrieved from the agent's queries.
        multi_papers (Dict[str, Any]): Stores multiple recommended papers from various sources.
        article_data (Dict[str, Any]): Stores the papers retrieved from Zotero and the pdf
        download agent with their metadata.
        zotero_write_approval_status (Dict[str, Any]): Stores the approval status and collection
        path for Zotero save operations.
        llm_model (BaseChatModel): The language model instance used for generating responses.
        text_embedding_model (Embeddings): The text embedding model used for
        similarity calculations.
    """

    # Agent state fields
    last_displayed_papers: Annotated[Dict[str, Any], replace_dict]
    papers: Annotated[Dict[str, Any], replace_dict]
    multi_papers: Annotated[Dict[str, Any], replace_dict]
    article_data: Annotated[Dict[str, Any], replace_dict]
    zotero_write_approval_status: Annotated[Dict[str, Any], replace_dict]
    llm_model: BaseChatModel
    text_embedding_model: Embeddings
