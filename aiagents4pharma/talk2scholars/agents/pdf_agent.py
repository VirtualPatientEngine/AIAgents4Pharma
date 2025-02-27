#!/usr/bin/env python3
"""
Agent for interacting with PDF documents via QnA.
"""

import logging
import hydra
from omegaconf import OmegaConf
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.pdf.qna import qna_tool
from ..tools.s2.query_results import query_results

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model="gpt-4o-mini"):
    """
    Returns the LangGraph app for PDF QnA.
    """
    # Load configuration using Hydra.
    with hydra.initialize(
        version_base=None,
        config_path="../../configs/agents/talk2scholars/pdf_agent"
    ):
        cfg = hydra.compose(config_name="default")
        logger.info("Loaded pdf_agent configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Use the configuration to set the model if none is provided.
    if llm_model is None:
        llm_model = cfg.openai_llms[0] if cfg.openai_llms else "gpt-4o-mini"

    def agent_pdf_node(state: Talk2Scholars):
        """
        Node that invokes the model with the current state.
        """
        logger.info("Creating Agent_PDF node with thread_id %s", uniq_id)
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return response

    # Define the tool node that includes our qna tool.
    tools = ToolNode([qna_tool, query_results])

    # Initialize the OpenAI model using configuration values.
    logger.info("Using OpenAI model %s", llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Create the agent.
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=cfg.pdf_agent,
        checkpointer=MemorySaver(),
    )

    # Define a new workflow graph with our state schema.
    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_pdf", agent_pdf_node)
    workflow.add_edge(START, "agent_pdf")

    # Initialize memory to persist state between runs.
    checkpointer = MemorySaver()

    # Compile the graph into a runnable app.
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("Compiled the PDF QnA agent graph.")

    return app
