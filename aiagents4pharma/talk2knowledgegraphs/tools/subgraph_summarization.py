"""
Tool for performing subgraph summarization.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import BaseTool
from langgraph.types import Command
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
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(description="Tool call ID.")
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")

class SubgraphSummarizationTool(BaseTool):
    """
    This tool performs subgraph summarization over textualized graph to highlight the most
    important information in responding to user's prompt.
    """
    name: str = "subgraph_summarization"
    description: str = "A tool to perform subgraph summarization over textualized graph."
    args_schema: Type[BaseModel] = SubgraphSummarizationInput

    def _run(self,
             tool_call_id: Annotated[str, InjectedToolCallId],
             state: Annotated[dict, InjectedState], prompt: str):
        """
        Run the subgraph summarization tool.

        Args:
            tool_call_id: The tool call ID.
            state: The injected state.
            prompt: The prompt to interact with the backend.
        """
        # Load hydra configuration
        logger.log(logging.INFO, "Loading Hydra configuration for subgraph summarization")
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name='config',
                overrides=['tools/subgraph_summarization=default']
            )
            cfg = cfg.tools.subgraph_summarization

        # Load the textualized subgraph
        textualized_subgraph = state["graph_text"]

        # Prepare prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", cfg.prompt_subgraph_summarization),
                ("human", "{input}"),
            ]
        )

        # Retrieve LLM from the state
        llm = state["llm_model"]

        # Prepare chain
        chain = prompt_template | llm | StrOutputParser()

        # Return the subgraph and textualized graph as JSON response
        response = chain.invoke({
            "input": prompt,
            "textualized_subgraph": textualized_subgraph,
        })

        return Command(
            update={
                "graph_summary": response,
                "messages":[
                    ToolMessage(
                        content=response,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
