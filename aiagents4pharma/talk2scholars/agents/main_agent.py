#!/usr/bin/env python3

"""
Main agent for the talk2scholars app.
"""

import logging
from typing import Literal, Any
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from ..agents import s2_agent
from ..state.state_talk2scholars import Talk2Scholars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_supervisor_node(llm: BaseChatModel, cfg: Any) -> str:
    """
    Creates a supervisor node following LangGraph patterns.

    Args:
        llm (BaseChatModel): The language model to use for generating responses.
        cfg (Any): The configuration object.

    Returns:
        str: The supervisor node function.
    """
    def supervisor_node(
        state: Talk2Scholars,
    ) -> Command[Literal["s2_agent", "__end__"]]:
        """
        Supervisor node that routes to appropriate sub-agents.

        Args:
            state (Talk2Scholars): The current state of the conversation.

        Returns:
            Command[Literal["s2_agent", "__end__"]]: The command to execute next.
        """
        logger.info(
            "Supervisor node called - Messages count: %d, Current Agent: %s",
            len(state["messages"]),
            state.get("current_agent", "None"),
        )

        messages = [{"role": "system", "content": cfg.state_modifier}] + state[
            "messages"
        ]
        response = llm.invoke(messages)
        goto = (
            "FINISH"
            if not any(
                kw in state["messages"][-1].content.lower()
                for kw in ["search", "paper", "find"]
            )
            else "s2_agent"
        )

        if goto == "FINISH":
            return Command(
                goto=END,
                update={
                    "messages": state["messages"]
                    + [AIMessage(content=response.content)],
                    "is_last_step": True,
                    "current_agent": None,
                },
            )

        return Command(
            goto="s2_agent",
            update={
                "messages": state["messages"],
                "is_last_step": False,
                "current_agent": "s2_agent",
            },
        )

    return supervisor_node


def get_app(thread_id: str, llm_model="gpt-4o-mini") -> StateGraph:
    """
    Returns the langraph app with hierarchical structure.

    Args:
        thread_id (str): The thread ID for the conversation.

    Returns:
        The compiled langraph app.
    """

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars main agent.")
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.main_agent
        logger.info("Hydra configuration loaded with values: %s", cfg)

    def call_s2_agent(state: Talk2Scholars) -> Command[Literal["__end__"]]:
        """
        Node for calling the S2 agent.

        Args:
            state (Talk2Scholars): The current state of the conversation.

        Returns:
            Command[Literal["__end__"]]: The command to execute next.
        """
        logger.info("Calling S2 agent with state: %s", state)
        app = s2_agent.get_app(thread_id, llm_model)
        response = app.invoke(state)
        logger.info("S2 agent completed with response: %s", response)
        return Command(
            goto=END,
            update={
                "messages": response["messages"],
                "papers": response.get("papers", []),
                "is_last_step": True,
                "current_agent": "s2_agent",
            },
        )

    logger.log(
        logging.INFO,
        "Using OpenAI model %s with temperature %s",
        llm_model,
        cfg.temperature
    )
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)
    workflow = StateGraph(Talk2Scholars)

    supervisor = make_supervisor_node(llm, cfg)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)

    # Define edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", END)

    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
