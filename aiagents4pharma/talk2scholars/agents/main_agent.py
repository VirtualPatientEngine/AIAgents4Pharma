#!/usr/bin/env python3

"""
Main agent for the talk2scholars app using ReAct pattern.

This module implements a hierarchical agent system where a supervisor agent
routes queries to specialized sub-agents. It follows the LangGraph patterns
for multi-agent systems and implements proper state management.

The main components are:
1. Supervisor node with ReAct pattern for intelligent routing.
2. S2 agent node for handling academic paper queries.
3. Shared state management via Talk2Scholars.
4. Hydra-based configuration system.

Example:
    app = get_app("thread_123", "gpt-4o-mini")
    result = app.invoke({
        "messages": [("human", "Find papers about AI agents")]
    })
"""

import logging
from typing import Literal, Callable
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from ..agents import s2_agent
from ..state.state_talk2scholars import Talk2Scholars

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hydra_config():
    """
    Loads and returns the Hydra configuration for the main agent.

    This function fetches the configuration settings for the Talk2Scholars
    agent, ensuring that all required parameters are properly initialized.

    Returns:
        Any: The configuration object for the main agent.
    """
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
    return cfg.agents.talk2scholars.main_agent


def make_supervisor_node(llm: BaseChatModel, thread_id: str) -> Callable:
    """
    Creates and returns a supervisor node for intelligent routing using the ReAct pattern.

    This function initializes a supervisor agent that processes user queries and
    determines the appropriate sub-agent for further processing. It applies structured
    reasoning to manage conversations and direct queries based on context.

    Args:
        llm (BaseChatModel): The language model used by the supervisor agent.
        thread_id (str): Unique identifier for the conversation session.

    Returns:
        Callable: A function that acts as the supervisor node in the LangGraph workflow.

    Example:
        supervisor = make_supervisor_node(llm, "thread_123")
        workflow.add_node("supervisor", supervisor)
    """
    logger.info("Loading Hydra configuration for Talk2Scholars main agent.")
    cfg = get_hydra_config()
    logger.info("Hydra configuration loaded with values: %s", cfg)

    # Create the supervisor agent using the main agent's configuration
    supervisor_agent = create_react_agent(
        llm,
        tools=[],  # Will add sub-agents later
        state_modifier=cfg.main_agent,
        state_schema=Talk2Scholars,
        checkpointer=MemorySaver(),
    )

    def supervisor_node(
        state: Talk2Scholars,
    ) -> Command[Literal["s2_agent", "__end__"]]:
        """
        Processes user queries and determines the next step in the conversation flow.

        This function examines the conversation state and decides whether to forward
        the query to a specialized sub-agent (e.g., S2 agent) or conclude the interaction.

        Args:
            state (Talk2Scholars): The current state of the conversation, containing
                messages, papers, and metadata.

        Returns:
            Command: The next action to be executed, along with updated state data.

        Example:
            result = supervisor_node(current_state)
            next_step = result.goto
        """
        logger.info(
            "Supervisor node called - Messages count: %d",
            len(state["messages"]),
        )

        # Invoke the supervisor agent with configurable thread_id
        result = supervisor_agent.invoke(
            state, {"configurable": {"thread_id": thread_id}}
        )
        goto = "s2_agent"
        logger.info("Supervisor agent completed with result: %s", result)

        return Command(goto=goto)

    return supervisor_node


def get_app(thread_id: str, llm_model: str = "gpt-4o-mini") -> StateGraph:
    """
    Initializes and returns the LangGraph application with a hierarchical agent system.

    This function sets up the full agent architecture, including the supervisor
    and sub-agents, and compiles the LangGraph workflow for handling user queries.

    Args:
        thread_id (str): Unique identifier for the conversation session.
        llm_model (str, optional): The language model to be used. Defaults to "gpt-4o-mini".

    Returns:
        StateGraph: A compiled LangGraph application ready for query invocation.

    Example:
        app = get_app("thread_123")
        result = app.invoke(initial_state)
    """
    cfg = get_hydra_config()

    def call_s2_agent(
        state: Talk2Scholars,
    ) -> Command[Literal["supervisor", "__end__"]]:
        """
        Calls the Semantic Scholar (S2) agent to process academic paper queries.

        This function invokes the S2 agent, retrieves relevant research papers,
        and updates the conversation state accordingly.

        Args:
            state (Talk2Scholars): The current conversation state, including user queries
                and any previously retrieved papers.

        Returns:
            Command: The next action to execute, along with updated messages and papers.

        Example:
            result = call_s2_agent(current_state)
            next_step = result.goto
        """
        logger.info("Calling S2 agent with state: %s", state)
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
        logger.info("S2 agent completed with response: %s", response)

        return Command(
            goto=END,
            update={
                "messages": response["messages"],
                "papers": response.get("papers", {}),
                "multi_papers": response.get("multi_papers", {}),
            },
        )

    # Initialize LLM
    logger.info("Using OpenAI model %s with temperature %s", llm_model, cfg.temperature)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Build the graph
    workflow = StateGraph(Talk2Scholars)
    supervisor = make_supervisor_node(llm, thread_id)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", END)

    # Compile the graph without initial state
    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
