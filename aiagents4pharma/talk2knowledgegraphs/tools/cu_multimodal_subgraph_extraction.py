"""
Tool for performing multimodal subgraph extraction.
"""

from typing import Type, Annotated
import logging
import numpy as np
import cudf
import cupy as cp
import hydra
import networkx as nx
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
import torch
from torch_geometric.data import Data
from ..utils.extractions.cu_multimodal_pcst import MultimodalPCSTPruning
from ..utils.embeddings.ollama import EmbeddingWithOllama
from .load_arguments import ArgumentData

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalSubgraphExtractionInput(BaseModel):
    """
    MultimodalSubgraphExtractionInput is a Pydantic model representing an input
    for extracting a subgraph.

    Args:
        prompt: Prompt to interact with the backend.
        tool_call_id: Tool call ID.
        state: Injected state.
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


class MultimodalSubgraphExtractionTool(BaseTool):
    """
    This tool performs subgraph extraction based on user's prompt by taking into account
    the top-k nodes and edges.
    """

    name: str = "subgraph_extraction"
    description: str = "A tool for subgraph extraction based on user's prompt."
    args_schema: Type[BaseModel] = MultimodalSubgraphExtractionInput

    def _prepare_query_modalities(self,
                                  prompt_emb: list,
                                  state: Annotated[dict, InjectedState],
                                  graph_nodes: cudf.DataFrame) -> cudf.DataFrame:
        """
        Prepare the modality-specific query for subgraph extraction.

        Args:
            prompt_emb: The embedding of the user prompt in a list.
            state: The injected state for the tool.
            graph_nodes: The nodes dataframe in the graph.

        Returns:
            A DataFrame containing the query embeddings and modalities.
        """
        # Initialize dataframes
        logger.log(logging.INFO, "Initializing dataframes")
        multimodal_df = cudf.DataFrame({"name": [], "node_type": []})
        query_df = cudf.DataFrame({"node_id": [],
                                    "node_type": [],
                                    "x": [],
                                    "desc_x": [],
                                    "use_description": []})

        # Loop over the uploaded files and find multimodal files
        logger.log(logging.INFO, "Looping over uploaded files")
        for i in range(len(state["uploaded_files"])):
            # Check if multimodal file is uploaded
            if state["uploaded_files"][i]["file_type"] == "multimodal":
                # Read the csv file
                multimodal_df = cudf.read_csv(state["uploaded_files"][i]["file_path"])

        # Check if the multimodal_df is empty
        logger.log(logging.INFO, "Checking if multimodal_df is empty")
        if len(multimodal_df) > 0:
            # Prepare multimodal_df
            logger.log(logging.INFO, "Preparing multimodal_df")
            multimodal_df.rename(columns={"name": "q_node_name",
                                          "node_type": "q_node_type"}, inplace=True)

            # Make and process a query dataframe by merging the graph_df and multimodal_df
            logger.log(logging.INFO, "Processing query dataframe")
            query_df = graph_nodes[
                ['node_id', 'node_name', 'node_type', 'enriched_node', 'x', 'desc', 'desc_x']
            ].merge(multimodal_df, how='cross')
            logger.log(logging.INFO, "Lowering case for node names (q_node_name)")
            query_df['q_node_name'] = query_df['q_node_name'].str.lower()
            logger.log(logging.INFO, "Lowering case for node names (node_name)")
            query_df['node_name'] = query_df['node_name'].str.lower()
            # Get the mask for filtering based on the query
            logger.log(logging.INFO, "Filtering based on the query")
            mask = (
                query_df['node_name'].str.contains(query_df['q_node_name']) &
                (query_df['node_type'] == query_df['q_node_type'])
            )
            query_df = query_df[mask]
            query_df = query_df[['node_id',
                                 'node_type', 
                                 'enriched_node', 
                                 'x', 
                                 'desc', 
                                 'desc_x']].reset_index(drop=True)
            query_df['use_description'] = False # set to False for modal-specific embeddings

            # Update the state by adding the the selected node IDs
            logger.log(logging.INFO, "Updating state with selected node IDs")
            state["selections"] = query_df.to_pandas().groupby(
                "node_type"
            )["node_id"].apply(list).to_dict()

        # Append a user prompt to the query dataframe
        logger.log(logging.INFO, "Adding user prompt to query dataframe")
        query_df = cudf.concat([
            query_df,
            cudf.DataFrame({
                'node_id': 'user_prompt',
                'node_type': 'prompt',
                # 'enriched_node': prompt,
                'x': prompt_emb,
                # 'desc': prompt,
                'desc_x': prompt_emb,
                'use_description': True # set to True for user prompt embedding
            })
        ]).reset_index(drop=True)

        return query_df

    def _perform_subgraph_extraction(self,
                                     state: Annotated[dict, InjectedState],
                                     cfg: dict,
                                     graph: dict,
                                     query_df: cudf.DataFrame) -> dict:
        """
        Perform multimodal subgraph extraction based on modal-specific embeddings.

        Args:
            state: The injected state for the tool.
            cfg: The configuration dictionary.
            graph: The graph dictionary.
            query_df: The DataFrame containing the query embeddings and modalities.

        Returns:
            A dictionary containing the extracted subgraph with nodes and edges.
        """
        # Initialize the subgraph dictionary
        subgraphs = {}
        subgraphs["nodes"] = []
        subgraphs["edges"] = []

        # Loop over query embeddings and modalities
        for q in query_df.to_pandas().iterrows():
            logger.log(logging.INFO, f"Processing query {q[1]['node_id']}")
            # Prepare the PCSTPruning object and extract the subgraph
            # Parameters were set in the configuration file obtained from Hydra
            subgraph = MultimodalPCSTPruning(
                topk=state["topk_nodes"],
                topk_e=state["topk_edges"],
                cost_e=cfg.cost_e,
                c_const=cfg.c_const,
                root=cfg.root,
                num_clusters=cfg.num_clusters,
                pruning=cfg.pruning,
                verbosity_level=cfg.verbosity_level,
                use_description=q[1]['use_description'],
            ).extract_subgraph(graph,
                               cp.array(q[1]['desc_x']).reshape(1, -1).astype(cp.float32),
                               cp.array(q[1]['x']).reshape(1, -1).astype(cp.float32),
                               q[1]['node_type'])

            # Append the extracted subgraph to the dictionary
            subgraphs["nodes"].append(subgraph["nodes"].tolist())
            subgraphs["edges"].append(subgraph["edges"].tolist())

        # Concatenate and get unique node and edge indices
        subgraphs["nodes"] = np.unique(
            np.concatenate([np.array(list_) for list_ in subgraphs["nodes"]])
        )
        subgraphs["edges"] = np.unique(
            np.concatenate([np.array(list_) for list_ in subgraphs["edges"]])
        )

        return subgraphs

    def _prepare_final_subgraph(self,
                                state:Annotated[dict, InjectedState],
                                subgraph: dict,
                                graph: dict,
                                cfg) -> dict:
        """
        Prepare the subgraph based on the extracted subgraph.

        Args:
            state: The injected state for the tool.
            subgraph: The extracted subgraph.
            graph: The graph dictionary.
            cfg: The configuration dictionary.

        Returns:
            A dictionary containing the PyG graph, NetworkX graph, and textualized graph.
        """
        # Convert the dict to a cudf DataFrame
        node_colors = {n: cfg.node_colors_dict[k]
                        for k, v in state["selections"].items() for n in v}
        color_df = cudf.DataFrame(list(node_colors.items()), columns=["node_id", "color"])

        # Prepare graph dataframes
        # Nodes
        graph_nodes = graph["nodes"].copy()
        graph_nodes = graph_nodes.iloc[subgraph['nodes']][
            ['node_id', 'node_name', 'node_type', 'desc', 'enriched_node']
        ]
        graph_nodes = graph_nodes.merge(color_df, on="node_id", how="left")
        graph_nodes['color'].fillna('black', inplace=True) # set default node color to black
        # Edges
        graph_edges = graph["edges"].copy()
        graph_edges = graph_edges.iloc[subgraph['edges']][
            ['head_id', 'tail_id', 'edge_type']
        ]

        # Prepare lists for visualization
        graph_dict = {}
        graph_dict["nodes"] = [(
            row.node_id,
            {'hover': "Node Name : " + row.node_name + "\n" +\
                "Node Type : " + row.node_type + "\n" +
                'Desc : ' + row.desc,
             'click': '$hover',
             'color': row.color})
             for row in graph_nodes.to_arrow().to_pandas().itertuples(index=False)]
        graph_dict["edges"] = [(
            row.head_id, 
            row.tail_id,
            {'label': tuple(row.edge_type)})
            for row in graph_edges.to_arrow().to_pandas().itertuples(index=False)]

        # Prepare the textualized subgraph
        graph_dict["text"] = (
            graph_nodes[
                ['node_id', 'desc']
            ].rename(columns={'desc': 'node_attr'}).to_arrow().to_pandas().to_csv(index=False)
            + "\n"
            + graph_edges[
                ['head_id', 'edge_type', 'tail_id']
            ].to_arrow().to_pandas().to_csv(index=False)
        )

        return graph_dict

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
        arg_data: ArgumentData = None,
    ) -> Command:
        """
        Run the subgraph extraction tool.

        Args:
            tool_call_id: The tool call ID for the tool.
            state: Injected state for the tool.
            prompt: The prompt to interact with the backend.
            arg_data (ArgumentData): The argument data.

        Returns:
            Command: The command to be executed.
        """
        logger.log(logging.INFO, "Invoking subgraph_extraction tool")

        # Load hydra configuration
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/multimodal_subgraph_extraction=default"]
            )
            cfg = cfg.tools.multimodal_subgraph_extraction

        # Retrieve source graph from the state
        initial_graph = {}
        initial_graph["source"] = state["dic_source_graph"][-1]  # The last source graph as of now
        # logger.log(logging.INFO, "Source graph: %s", source_graph)

        # Load the knowledge graph
        initial_graph["nodes"] = cudf.read_parquet(initial_graph["source"]["kg_nodes_path"])
        initial_graph["edges"] = cudf.read_parquet(initial_graph["source"]["kg_edges_path"])

        # Prepare the query embeddings and modalities
        logger.log(logging.INFO, "_prepare_query_modalities")
        query_df = self._prepare_query_modalities(
            [EmbeddingWithOllama(model_name=cfg.ollama_embeddings[0]).embed_query(prompt)],
            state,
            initial_graph["nodes"]
        )

        # Perform subgraph extraction
        logger.log(logging.INFO, "_perform_subgraph_extraction")
        subgraphs = self._perform_subgraph_extraction(state,
                                                      cfg,
                                                      initial_graph,
                                                      query_df)

        # Prepare subgraph as a NetworkX graph and textualized graph
        logger.log(logging.INFO, "_prepare_final_subgraph")
        final_subgraph = self._prepare_final_subgraph(state,
                                                      subgraphs,
                                                      initial_graph,
                                                      cfg)

        # Prepare the dictionary of extracted graph
        logger.log(logging.INFO, "dic_extracted_graph")
        dic_extracted_graph = {
            "name": arg_data.extraction_name,
            "tool_call_id": tool_call_id,
            "graph_source": initial_graph["source"]["name"],
            "topk_nodes": state["topk_nodes"],
            "topk_edges": state["topk_edges"],
            "graph_dict": {
                "nodes": final_subgraph["nodes"],
                "edges": final_subgraph["edges"],
            },
            "graph_text": final_subgraph["text"],
            "graph_summary": None,
        }

        # Prepare the dictionary of updated state
        dic_updated_state_for_model = {}
        for key, value in {
            "dic_extracted_graph": [dic_extracted_graph],
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        # Return the updated state of the tool
        return Command(
            update=dic_updated_state_for_model | {
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"Subgraph Extraction Result of {arg_data.extraction_name}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
