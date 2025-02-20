"""
Tool for performing named-entity recognition.
"""

from typing import Type, Annotated
import ast
import logging
import pickle
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
import hydra
from .load_arguments import ArgumentData

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NamedEntityRecognitionInput(BaseModel):
    """
    NamedEntityRecognitionInput is a Pydantic model representing an input for
    performing named-entity recognition.

    Args:
        tool_call_id: Tool call ID.
        state: Injected state.
        prompt: Prompt to interact with the backend.
        arg_data: Argument for analytical process over graph data.
    """

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        description="Tool call ID."
    )
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")
    prompt: str = Field(description="Prompt to interact with the backend.")
    arg_data: ArgumentData = Field(
        description="Experiment over graph data.", default=None
    )


class NamedEntityRecognitionTool(BaseTool):
    """
    This tool performs named-entity recognition given a user prompt.
    """

    name: str = "named_entity_recognition"
    description: str = (
        """A tool to perform named-entity recognition given a user prompt."""
    )
    args_schema: Type[BaseModel] = NamedEntityRecognitionInput

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
        arg_data: ArgumentData = None,
    ):
        """
        Run the subgraph summarization tool.

        Args:
            tool_call_id: The tool call ID.
            state: The injected state.
            prompt: The prompt to interact with the backend.
            arg_data: The argument data.
        """
        logger.log(logging.INFO, "Invoking named-entity recognition")

        # Load hydra configuration
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name="config",
                overrides=["tools/named_entity_recognition=default"],
            )
            cfg = cfg.tools.named_entity_recognition

        # Retrieve source graph from the state
        graph_data = {}
        graph_data["source"] = state["dic_source_graph"][-1]  # The last source graph as of now
        # logger.log(logging.INFO, "Source graph: %s", graph_data["source"])

        # Load the knowledge graph
        with open(graph_data["source"]["kg_pyg_path"], "rb") as f:
            graph_data["pyg"] = pickle.load(f)

        # Construct a pandas dataframe over nodes
        graph_data["nodes_df"] = pd.DataFrame(
            {
                "node_id": graph_data["pyg"].node_id,
                # "node_name": graph_data["pyg"].node_name,
                # "node_type": graph_data["pyg"].node_type,
            }
        )
        logger.log(
            logging.INFO, "Shape of the dataframe: %s", graph_data["nodes_df"].shape
        )

        # Create a pandas dataframe agent
        df_agent = create_pandas_dataframe_agent(
            state["llm_model"],
            allow_dangerous_code=True,
            agent_type="tool-calling",
            df=graph_data["nodes_df"],
            max_iterations=5,
            include_df_in_prompt=True,
            number_of_head_rows=graph_data["nodes_df"].shape[0],
            verbose=True,
            prefix=cfg.prompt_ner,
        )

        # Invoke the agent with the given prompt
        response = df_agent.invoke(prompt, stream_mode=None)
        # logger.log(logging.INFO, "Response: %s", response)

        # Prepare the dictionary of extracted graph
        dic_extracted_graph = {
            "name": arg_data.extraction_name,
            "tool_call_id": tool_call_id,
            "graph_source": graph_data["source"]["name"],
            "ner_nodes": ast.literal_eval(response["output"]),
            "topk_nodes": None,
            "topk_edges": None,
            "graph_dict": None,
            "graph_text": None,
            "graph_summary": None,
        }
        # logger.log(logging.INFO, "Extracted graph: %s", dic_extracted_graph)

        # Prepare the dictionary of updated state
        dic_updated_state_for_model = {}
        for key, value in {
            "dic_extracted_graph": [dic_extracted_graph],
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        # Return the updated state of the tool
        return Command(
            update=dic_updated_state_for_model
            | {
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"NER Result of {arg_data.extraction_name}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
