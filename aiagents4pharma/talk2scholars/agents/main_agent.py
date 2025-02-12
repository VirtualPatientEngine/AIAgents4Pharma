#!/usr/bin/env python3

"""
Main agent for the talk2scholars app using ReAct pattern.

This module implements a hierarchical agent system where a supervisor agent
routes queries to specialized sub-agents. It follows the LangGraph patterns
for multi-agent systems and implements proper state management.

The main components are:
1. Supervisor node with ReAct pattern for intelligent routing
2. S2 agent node for handling academic paper queries
3. Shared state management via Talk2Scholars
4. Hydra-based configuration system

Example:
    app = get_app("thread_123", "gpt-4o-mini")
    result = app.invoke({
        "messages": [("human", "Find papers about AI agents")]
    })
"""

import logging
from typing import Any, Literal, Callable
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


def make_supervisor_node(llm: BaseChatModel, cfg: Any, thread_id: str) -> Callable:
    """
    Creates and returns a supervisor node for intelligent routing using the ReAct pattern.

    This function initializes a supervisor agent that processes user queries and 
    determines the appropriate sub-agent for further processing. It applies structured 
    reasoning to manage conversations and direct queries based on context.

    Args:
        llm (BaseChatModel): The language model used by the supervisor agent.
        cfg (Any): Configuration object containing system prompts and settings.
        thread_id (str): Unique identifier for the conversation session.

    Returns:
        Callable: A function that acts as the supervisor node in the LangGraph workflow.

    Example:
        supervisor = make_supervisor_node(llm, cfg, "thread_123")
        workflow.add_node("supervisor", supervisor)
    """
    # Load hydra configuration
    logger.info("Load Hydra configuration for Talk2Scholars main agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.main_agent
        logger.info("Hydra configuration loaded with values: %s", cfg)
    # Create the supervisor agent using config's main_agent prompt
    supervisor_agent = create_react_agent(
        llm,
        tools=[],  # Will make tool of s2agent later
        # prompt=cfg.main_agent,
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
            "Supervisor node called - Messages count: %d, Current Agent: %s",
            len(state["messages"]),
            state.get("current_agent", "None"),
        )

        # Invoke the supervisor agent with configurable thread_id
        result = supervisor_agent.invoke(
            state,
            {"configurable": {"thread_id": thread_id}},
        )
        decision = result["messages"][-1].content
        print("decision", decision)
        # goto = decision["next"]
        goto = "s2_agent"

        return Command(
            goto=goto,
            update={
                "messages": state["messages"],
                "papers": state.get("papers", {}),
                "multi_papers": state.get("multi_papers", {}),
            },
        )

    return supervisor_node


def get_app(
    thread_id: str, llm_model: str = "gpt-4o-mini", cfg: Any = None
) -> StateGraph:
    """
    Initializes and returns the LangGraph application with a hierarchical agent system.

    This function sets up the full agent architecture, including the supervisor 
    and sub-agents, and compiles the LangGraph workflow for handling user queries.

    Args:
        thread_id (str): Unique identifier for the conversation session.
        llm_model (str, optional): The language model to be used. Defaults to "gpt-4o-mini".
        cfg (Any, optional): Configuration object for customizing agent behavior.

    Returns:
        StateGraph: A compiled LangGraph application ready for query invocation.

    Example:
        app = get_app("thread_123")
        result = app.invoke(initial_state)
    """

    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.main_agent
        logger.info("Hydra configuration loaded with values: %s", cfg)

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

        # Pass initial state in invoke
        response = app.invoke(
            {
                **state,
                "papers": state.get("papers", {}),
                "multi_papers": state.get("multi_papers", {}),
            }
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
    logger.info(
        "Using OpenAI model %s with temperature %s",
        llm_model,
        cfg.temperature,
    )
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Build the graph
    workflow = StateGraph(Talk2Scholars)

    # Add nodes
    supervisor = make_supervisor_node(llm, cfg, thread_id)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)

    # Define edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", END)  # End the conversation

    # Compile the graph without initial state
    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
