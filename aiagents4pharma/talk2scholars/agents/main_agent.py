#!/usr/bin/env python3

"""
Main agent for the talk2scholars app using ReAct pattern.

This module implements a hierarchical agent system where a supervisor agent
routes queries to specialized sub-agents. It follows the LangGraph patterns
for multi-agent systems and implements proper state management.
"""

import logging
from typing import Literal, Callable
from pydantic import BaseModel
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from ..agents import s2_agent
from ..state.state_talk2scholars import Talk2Scholars

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hydra_config():
    """
    Loads the Hydra configuration for the main agent.

    This function initializes the Hydra configuration system and retrieves the settings
    for the `Talk2Scholars` agent, ensuring that all required parameters are loaded.

    Returns:
        DictConfig: The configuration object containing parameters for the main agent.
    """
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
    return cfg.agents.talk2scholars.main_agent


def make_supervisor_node(llm_model: BaseChatModel, thread_id: str) -> Callable:
    """
    Creates the supervisor node responsible for routing user queries to the appropriate sub-agents.

    This function initializes the routing logic by leveraging the system and router prompts defined
    in the Hydra configuration. The supervisor determines whether to
    call a sub-agent (like `s2_agent`)
    or directly generate a response using the language model.

    Args:
        llm_model (BaseChatModel): The language model used for decision-making.
        thread_id (str): Unique identifier for the current conversation session.

    Returns:
        Callable: The supervisor node function that processes user queries and
        decides the next step.
    """
    cfg = get_hydra_config()
    logger.info("Hydra configuration for Talk2Scholars main agent loaded: %s", cfg)
    members = ["s2_agent"]
    options = ["FINISH"] + members
    # Define system prompt for general interactions
    system_prompt = cfg.system_prompt
    # Define router prompt for routing to sub-agents
    router_prompt = cfg.router_prompt

    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(
        state: Talk2Scholars,
    ) -> Command:
        """
        Handles the routing logic for the supervisor agent.

        This function determines the next agent to invoke based on the router prompt response.
        If no further processing is required, it generates an AI response using the system prompt.

        Args:
            state (Talk2Scholars): The current conversation state, including messages
            exchanged so far.

        Returns:
            Command: A command dictating whether to invoke a sub-agent or generate a final response.
        """
        messages = [SystemMessage(content=router_prompt)] + state["messages"]
        structured_llm = llm_model.with_structured_output(Router)
        response = structured_llm.invoke(messages)
        goto = response.next
        logger.info("Routing to: %s, Thread ID: %s", goto, thread_id)
        if goto == "FINISH":
            goto = END  # Using END from langgraph.graph
            # If no agents were called, and the last message was
            # from the user, call the LLM to respond to the user
            # with a slightly different system prompt.
            if isinstance(messages[-1], HumanMessage):
                response = llm_model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                    ]
                    + messages[1:]
                )
                return Command(
                    goto=goto, update={"messages": AIMessage(content=response.content)}
                )
        # Go to the requested agent
        return Command(goto=goto)

    return supervisor_node


def get_app(
    thread_id: str,
    llm_model: BaseChatModel = ChatOpenAI(model="gpt-4o-mini", temperature=0),
):
    """
    Initializes and returns the LangGraph-based hierarchical agent system.

    This function constructs the agent workflow by defining nodes for the supervisor
    and sub-agents. It compiles the graph using `StateGraph` to enable structured
    conversational workflows.

    Args:
        thread_id (str): A unique session identifier for tracking conversation state.
        llm_model (BaseChatModel, optional): The language model used for query processing.
            Defaults to `ChatOpenAI(model="gpt-4o-mini", temperature=0)`.

    Returns:
        StateGraph: A compiled LangGraph application that can process user queries.

    Example:
        >>> app = get_app("thread_123")
        >>> result = app.invoke(initial_state)
    """
    cfg = get_hydra_config()

    def call_s2_agent(
        state: Talk2Scholars,
    ) -> Command[Literal["supervisor"]]:
        """
        Invokes the Semantic Scholar (S2) agent to retrieve relevant research papers.

        This function calls the `s2_agent` and updates the conversation state with retrieved
        academic papers. The agent uses Semantic Scholar's API to find papers based on
        user queries.

        Args:
            state (Talk2Scholars): The current state of the conversation, containing messages
                and any previous search results.

        Returns:
            Command: A command to update the conversation state with the retrieved papers
                and return control to the supervisor node.

        Example:
            >>> result = call_s2_agent(current_state)
            >>> next_step = result.goto
        """
        logger.info("Calling S2 agent")
        app = s2_agent.get_app(thread_id, llm_model)

        # Invoke the S2 agent, passing state,
        # Pass both config_id and thread_id
        response = app.invoke(
            state,
            {
                "configurable": {
                    "config_id": thread_id,
                    "thread_id": thread_id,
                }
            },
        )
        logger.info("S2 agent completed with response")
        return Command(
            update={
                "messages": response["messages"],
                "papers": response.get("papers", {}),
                "multi_papers": response.get("multi_papers", {}),
                "last_displayed_papers": response.get("last_displayed_papers", {}),
            },
            # Always return to supervisor
            goto="supervisor",
        )

    # Initialize LLM
    logger.info("Using model %s with temperature %s", llm_model, cfg.temperature)

    # Build the graph
    workflow = StateGraph(Talk2Scholars)
    supervisor = make_supervisor_node(llm_model, thread_id)
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)
    # Add edges
    workflow.add_edge(START, "supervisor")
    # Compile the workflow
    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
