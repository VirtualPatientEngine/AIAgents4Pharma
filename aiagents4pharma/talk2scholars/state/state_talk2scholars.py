"""
State management for the Talk2Scholars agent.

This module defines the state class `Talk2Scholars`, which maintains the conversation
context, retrieved papers, and other relevant metadata. The state ensures consistency
across agent interactions.
"""

import logging
from collections.abc import Mapping
from typing import Annotated, Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_dict(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges the existing dictionary with a new dictionary.

    This function logs the state merge and ensures that the new values
    are appended to the existing state without overwriting other entries.
    Args:
        existing (Dict[str, Any]): The current dictionary state.
        new (Dict[str, Any]): The new dictionary state to merge.
    Returns:
        Dict[str, Any]: The merged dictionary state.
    """
    merged = dict(existing) if existing else {}
    merged.update(new or {})
    return merged


def replace_dict(existing: Dict[str, Any], new: Any) -> Any:
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
    # If new is not a mapping, just replace existing value outright
    if not isinstance(new, Mapping):
        return new
    # In-place replace: clear existing mapping and update with new entries
    existing.clear()
    existing.update(new)
    return existing


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
    # Key controlling UI display: always replace to reference latest output
    # Stores the most recently displayed papers metadata
    last_displayed_papers: Annotated[Dict[str, Any], replace_dict]
    # Accumulative keys: merge new entries into existing state
    papers: Annotated[Dict[str, Any], merge_dict]
    multi_papers: Annotated[Dict[str, Any], merge_dict]
    article_data: Annotated[Dict[str, Any], merge_dict]
    # Approval status: always replace to reflect latest operation
    zotero_write_approval_status: Annotated[Dict[str, Any], replace_dict]
    llm_model: BaseChatModel
    text_embedding_model: Embeddings
