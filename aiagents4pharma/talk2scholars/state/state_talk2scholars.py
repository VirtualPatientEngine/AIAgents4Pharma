"""
This is the state file for the talk2scholars agent.
"""

import logging
from typing import Annotated, Any, Dict
from langgraph.prebuilt.chat_agent_executor import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_dict(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the existing dict with the new one."""
    logger.info("Updating existing state %s with the state dict: %s", existing, new)
    return new


class Talk2Scholars(AgentState):
    """
    The state for the talk2scholars agent, inheriting from AgentState.

    Attributes:
        papers: Dictionary of papers from search results
        multi_papers: Dictionary of papers from multi-paper recommendations
        llm_model: Model being used
    """

    # Agent state fields
    papers: Annotated[Dict[str, Any], replace_dict]
    multi_papers: Annotated[Dict[str, Any], replace_dict]
    llm_model: str
