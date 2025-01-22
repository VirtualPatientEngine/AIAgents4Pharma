"""
This is the state file for the talk2comp agent.
"""

import logging
from typing import Annotated, List, Optional
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import NotRequired, Required

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def replace_list(existing: List[str], new: List[str]) -> List[str]:
    """Replace the existing list with the new one."""
    logger.info("Updating existing state %s with the state list: %s",
                existing,
                new)
    return new


class Talk2Competitors(AgentState):
    """
    The state for the talk2comp agent, inheriting from AgentState.
    """
    papers: Annotated[List[str], replace_list]
    search_table: NotRequired[str]
    next: str  # Required for routing in LangGraph
    current_agent: NotRequired[Optional[str]]
    is_last_step: Required[bool]  # Required field for LangGraph
    llm_model: str
