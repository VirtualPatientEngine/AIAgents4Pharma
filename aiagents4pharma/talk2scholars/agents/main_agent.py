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
from langchain_core.messages import AIMessage
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


def make_supervisor_node(llm: BaseChatModel, cfg: Any) -> Callable:
    """
    Creates a supervisor node using ReAct pattern for intelligent routing.

    This function creates a supervisor agent that can make informed decisions
    about routing queries to appropriate sub-agents. It uses the ReAct pattern
    for structured reasoning and decision making.

    Args:
        llm (BaseChatModel): The language model to use for the supervisor
        cfg (Any): Configuration object containing system prompts and settings

    Returns:
        Callable: A function that can be used as a node in the StateGraph

    Example:
        supervisor = make_supervisor_node(llm, cfg)
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
        tools=[],  # Supervisor doesn't need tools, just makes routing decisions
        prompt=cfg.main_agent,
        state_schema=Talk2Scholars,
        checkpointer=MemorySaver(),
    )

    def supervisor_node(
        state: Talk2Scholars,
    ) -> Command[Literal["s2_agent", "__end__"]]:
        """
        Supervisor node that routes to appropriate sub-agents using ReAct pattern.

        This function takes the current state of the conversation and determines
        the next action to take, either routing to a sub-agent or finishing
        the conversation.

        Args:
            state (Talk2Scholars): Current conversation state containing
                messages, papers, and other relevant information

        Returns:
            Command: Next action and state updates to be performed

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
            {"configurable": {"thread_id": state.get("thread_id")}},
        )
        decision = result["messages"][-1].content
        goto = decision["next"]

        if goto == "FINISH":
            return Command(
                goto=END,
                update={
                    "messages": state["messages"]
                    + [AIMessage(content=result["messages"][-1].content)],
                    "is_last_step": True,
                    "current_agent": None,
                    "papers": state.get("papers", {}),
                    "multi_papers": state.get("multi_papers", {}),
                    "need_search": False,
                },
            )

        return Command(
            goto=goto,
            update={
                "messages": state["messages"],
                "is_last_step": False,
                "current_agent": goto,
                "papers": state.get("papers", {}),
                "multi_papers": state.get("multi_papers", {}),
                "thread_id": state.get("thread_id"),
                "need_search": state.get("need_search", False),
            },
        )

    return supervisor_node


def get_app(
    thread_id: str, llm_model: str = "gpt-4o-mini", cfg: Any = None
) -> StateGraph:
    """
    Returns the compiled LangGraph application with hierarchical structure.

    This function creates and configures the complete agent system, including
    the supervisor and all sub-agents. It handles configuration loading,
    state initialization, and graph compilation.

    Args:
        thread_id (str): Unique identifier for the conversation thread
        llm_model (str, optional): Name of the LLM model to use.
            Defaults to "gpt-4o-mini"

    Returns:
        StateGraph: The compiled application ready for invocation

    Example:
        app = get_app("thread_123")
        result = app.invoke(initial_state)
    """

    def call_s2_agent(
        state: Talk2Scholars,
    ) -> Command[Literal["supervisor", "__end__"]]:
        """
        Node for calling the Semantic Scholar agent.

        This function handles the invocation of the S2 agent and processes
        its response, ensuring proper state updates and message passing.

        Args:
            state (Talk2Scholars): Current conversation state

        Returns:
            Command: Next action and state updates
        """
        logger.info("Calling S2 agent with state: %s", state)
        app = s2_agent.get_app(thread_id, llm_model, cfg)

        # Pass initial state in invoke
        response = app.invoke(
            {
                **state,
                "thread_id": thread_id,
                "papers": state.get("papers", {}),
                "multi_papers": state.get("multi_papers", {}),
                "is_last_step": False,
                "current_agent": "s2_agent",
                "need_search": state.get("need_search", False),
            }
        )
        logger.info("S2 agent completed with response: %s", response)

        return Command(
            goto="supervisor",  # Return to supervisor for next decision
            update={
                "messages": response["messages"],
                "papers": response.get("papers", {}),
                "multi_papers": response.get("multi_papers", {}),
                "is_last_step": False,
                "current_agent": "s2_agent",
                "thread_id": thread_id,
                "need_search": response.get("need_search", False),
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
    supervisor = make_supervisor_node(llm, cfg)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)

    # Define edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", "supervisor")  # Report back to supervisor

    # Compile the graph without initial state
    app = workflow.compile()
    logger.info("Main agent workflow compiled")
    return app
