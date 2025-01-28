"""
Tool for performing subgraph summarization.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
import hydra

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubgraphSummarizationInput(BaseModel):
    """
    SubgraphSummarizationInput is a Pydantic model representing an input for
    summarizing a given textualized subgraph.

    Args:
        prompt (str): Prompt to interact with the backend.
        textualized_subgraph (str): Textualized subgraph.
    """
    prompt: str = Field(description="Prompt to interact with the backend.")
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")

class SubgraphSummarizationTool(BaseTool):
    """
    This tool performs subgraph summarization over textualized graph to highlight the most
    important information in responding to user's prompt.
    """
    name: str = "subgraph_summarization"
    description: str = "A tool to perform subgraph summarization over textualized graph."
    args_schema: Type[BaseModel] = SubgraphSummarizationInput

    def _run(self, state: Annotated[dict, InjectedState], prompt: str):
        """
        Run the subgraph summarization tool.

        Args:
            state: The injected state.
            prompt: The prompt to interact with the backend.
        """
        # Load hydra configuration
        logger.log(logging.INFO, "Load Hydra configuration for subgraph summarization")
        with hydra.initialize(version_base=None, config_path="../../../../../configs"):
            cfg = hydra.compose(
                config_name='config',
                overrides=['web/backend/routers/tools/subgraph_summarization=default']
            )
            cfg = cfg.web.backend.routers.tools.subgraph_summarization

        # Load the textualized subgraph
        logger.log(logging.INFO, "Load the most recent extracted subgraph")
        textualized_subgraph = state["recent_subgraph"]

        # Prepare prompt template
        logger.log(logging.INFO, "Prepare prompt template")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", cfg.prompt_subgraph_summarization),
                ("human", "{input}"),
            ]
        )

        # Prepare LLM
        logger.log(logging.INFO, "Prepare LLM")
        if state["llm_model"] in cfg.openai_llms:
            llm = ChatOpenAI(state["llm_model"],
                            api_key=cfg.openai_api_key,
                            temperature=cfg.temperature,
                            streaming=cfg.streaming)
        else:
            llm = ChatOllama(model=state["llm_model"],
                            temperature=cfg.temperature,
                            streaming=cfg.streaming)

        # Prepare chain
        logger.log(logging.INFO, "Prepare chain")
        chain = prompt_template | llm | StrOutputParser()

        # Return the subgraph and textualized graph as JSON response
        logger.log(logging.INFO, "Return the ouput after invoking the chain")
        response = chain.invoke({
            "input": prompt,
            "model_name": state["llm_model"],
            "textualized_subgraph": textualized_subgraph,
        })

        return response
