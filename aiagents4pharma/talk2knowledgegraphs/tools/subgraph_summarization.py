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
        tool_call_id: Tool call ID.
        state: Injected state.
        prompt: Prompt to interact with the backend.
        extraction_name: Name assigned to the subgraph extraction process
    """

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        description="Tool call ID."
    )
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")
    prompt: str = Field(description="Prompt to interact with the backend.")
    extraction_name: str = Field(
        description="""Name assigned to the subgraph extraction process
                                 when the subgraph_extraction tool is invoked."""
    )


class SubgraphSummarizationTool(BaseTool):
    """
    This tool performs subgraph summarization over textualized graph to highlight the most
    important information in responding to user's prompt.
    """

    name: str = "subgraph_summarization"
    description: str = """A tool to perform subgraph summarization over textualized graph
                        for responding to user's follow-up prompt(s)."""
    args_schema: Type[BaseModel] = SubgraphSummarizationInput

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
        extraction_name: str,
    ):
        """
        Run the subgraph summarization tool.

        Args:
            tool_call_id: The tool call ID.
            state: The injected state.
            prompt: The prompt to interact with the backend.
            extraction_name: The name assigned to the subgraph extraction process.
        """
        logger.log(
            logging.INFO, "Invoking subgraph_summarization tool for %s", extraction_name
        )

        # Load hydra configuration
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/subgraph_summarization=default"]
            )
            cfg = cfg.tools.subgraph_summarization

        # Load the extracted graph
        extracted_graph = {dic["name"]: dic for dic in state["dic_extracted_graph"]}
        # logger.log(logging.INFO, "Extracted graph: %s", extracted_graph)

        # Prepare prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", cfg.prompt_subgraph_summarization),
                ("human", "{input}"),
            ]
        )

        # Prepare chain
        chain = prompt_template | state["llm_model"] | StrOutputParser()

        # Return the subgraph and textualized graph as JSON response
        response = chain.invoke(
            {
                "input": prompt,
                "textualized_subgraph": extracted_graph[extraction_name]["graph_text"],
            }
        )

        # Store the response as graph_summary in the extracted graph
        for key, value in extracted_graph.items():
            if key == extraction_name:
                value["graph_summary"] = response

        # Prepare the dictionary of updated state
        dic_updated_state_for_model = {}
        for key, value in {
            "dic_extracted_graph": list(extracted_graph.values()),
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        # Return the updated state of the tool
        return Command(
            update=dic_updated_state_for_model
            | {
                # update the message history
                "messages": [ToolMessage(content=response, tool_call_id=tool_call_id)]
            }
        )
