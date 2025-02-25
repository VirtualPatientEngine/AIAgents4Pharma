#!/usr/bin/env python3

"""
Agent for interacting with Zotero
"""

import logging
from typing import Any, Dict
import hydra
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.zotero.zotero_read import zotero_search_tool
from ..tools.s2.display_results import display_results as s2_display
from ..tools.s2.query_results import query_results as s2_query_results

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_app(
    uniq_id, llm_model: BaseChatModel = ChatOpenAI(model="gpt-4o-mini", temperature=0)
):
    """
    This function returns the langraph app.
    """

    # def agent_zotero_node(state: Talk2Scholars) -> Command[Literal["supervisor"]]:
    def agent_zotero_node(state: Talk2Scholars) -> Dict[str, Any]:
        """
        This function calls the model and always returns to supervisor.
        """
        logger.log(
            logging.INFO, "Creating Agent_Zotero node with thread_id %s", uniq_id
        )
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars Zotero agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["agents/talk2scholars/zotero_agent=default"],
        )
        cfg = cfg.agents.talk2scholars.zotero_agent

    # Define the tools
    tools = ToolNode([zotero_search_tool, s2_display, s2_query_results])

    # Define the model
    logger.log(logging.INFO, "Using model %s", llm_model)

    # Create the agent
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=cfg.zotero_agent,
        checkpointer=MemorySaver(),
    )

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_zotero", agent_zotero_node)
    workflow.add_edge(START, "agent_zotero")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO, "Compiled the graph")

    return app
