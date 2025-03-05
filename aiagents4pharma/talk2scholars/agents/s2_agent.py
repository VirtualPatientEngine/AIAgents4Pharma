# /usr/bin/env python3

"""
Agent for interacting with Semantic Scholar
"""

import logging
from typing import Any, Dict
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.s2.search import search_tool as s2_search
from ..tools.s2.display_results import display_results as s2_display
from ..tools.s2.query_results import query_results as s2_query_results
from ..tools.s2.retrieve_semantic_scholar_paper_id import (
    retrieve_semantic_scholar_paper_id as s2_retrieve_id,
)
from ..tools.s2.single_paper_rec import (
    get_single_paper_recommendations as s2_single_rec,
)
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations as s2_multi_rec

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_app(uniq_id, llm_model: BaseChatModel):
    """
    Initializes and returns the LangGraph application for the Semantic Scholar (S2) agent.

    This function sets up the S2 agent, which integrates various tools to search, retrieve,
    and display research papers from Semantic Scholar. The agent follows the ReAct pattern
    for structured interaction.

    Args:
        uniq_id (str): Unique identifier for the current conversation session.
        llm_model (BaseChatModel, optional): The language model to be used by the agent.
            Defaults to `ChatOpenAI(model="gpt-4o-mini", temperature=0)`.

    Returns:
        StateGraph: A compiled LangGraph application that enables the S2 agent to process
            user queries and retrieve research papers.

    Example:
        >>> app = get_app("thread_123")
        >>> result = app.invoke(initial_state)
    """

    # def agent_s2_node(state: Talk2Scholars) -> Command[Literal["supervisor"]]:
    def agent_s2_node(state: Talk2Scholars) -> Dict[str, Any]:
        """
        Processes the user query and retrieves relevant research papers.

        This function calls the language model using the configured `ReAct` agent to analyze
        the state and generate an appropriate response. The function then returns control
        to the main supervisor.

        Args:
            state (Talk2Scholars): The current conversation state, including messages exchanged
                and any previously retrieved research papers.

        Returns:
            Dict[str, Any]: A dictionary containing the updated conversation state.

        Example:
            >>> result = agent_s2_node(current_state)
            >>> papers = result.get("papers", [])
        """
        logger.log(logging.INFO, "Creating Agent_S2 node with thread_id %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    logger.log(logging.INFO, "thread_id, llm_model: %s, %s", uniq_id, llm_model)

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars S2 agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/s2_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.s2_agent
        logger.log(logging.INFO, "Loaded configuration for S2 agent")

    # Define the tools
    tools = ToolNode(
        [
            s2_search,
            s2_display,
            s2_query_results,
            s2_retrieve_id,
            s2_single_rec,
            s2_multi_rec,
        ]
    )

    # Define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)

    # Create the agent
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        prompt=cfg.s2_agent,
        checkpointer=MemorySaver(),
    )

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_s2", agent_s2_node)
    workflow.add_edge(START, "agent_s2")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer, name="agent_s2")
    logger.log(
        logging.INFO,
        "Compiled the graph with thread_id %s and llm_model %s",
        uniq_id,
        llm_model,
    )

    return app
