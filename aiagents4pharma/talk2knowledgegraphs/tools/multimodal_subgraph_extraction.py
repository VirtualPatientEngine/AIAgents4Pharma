"""
Tool for performing multimodal subgraph extraction.
"""

from typing import Type, Annotated
import logging
import pickle
import numpy as np
import pandas as pd
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
from ..utils.extractions.multimodal_pcst import MultimodalPCSTPruning
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
                                  pyg_graph: Data) -> pd.DataFrame:
        """
        Prepare the modality-specific query for subgraph extraction.

        Args:
            prompt_emb: The embedding of the user prompt in a list.
            state: The injected state for the tool.
            pyg_graph: The PyTorch Geometric graph Data.

        Returns:
            A DataFrame containing the query embeddings and modalities.
        """
        # Initialize dataframes
        multimodal_df = pd.DataFrame({"name": []})
        query_df = pd.DataFrame({"node_id": [],
                                 "node_type": [],
                                 "x": [],
                                 "desc_x": [],
                                 "use_description": []})

        # Loop over the uploaded files and find multimodal files
        for i in range(len(state["uploaded_files"])):
            # Check if multimodal file is uploaded
            if state["uploaded_files"][i]["file_type"] == "multimodal":
                # Read the Excel file
                multimodal_df = pd.read_excel(state["uploaded_files"][i]["file_path"],
                                              sheet_name=None)

        # Check if the multimodal_df is empty
        if len(multimodal_df) > 0:
            # Merge all obtained dataframes into a single dataframe
            multimodal_df = pd.concat(multimodal_df).reset_index()
            multimodal_df.drop(columns=["level_1"], inplace=True)
            multimodal_df.rename(columns={"level_0": "q_node_type",
                                        "name": "q_node_name"}, inplace=True)
            # Since an excel sheet name could not contain a `/`,
            # but the node type can be 'gene/protein' as exists in the PrimeKG
            multimodal_df["q_node_type"] = multimodal_df.q_node_type.apply(
                lambda x: x.replace('-', '/')
            )

            # Convert PyG graph to a DataFrame for easier filtering
            graph_df = pd.DataFrame({
                "node_id": pyg_graph.node_id,
                "node_name": pyg_graph.node_name,
                "node_type": pyg_graph.node_type,
                "x": pyg_graph.x,
                "desc_x": pyg_graph.desc_x.tolist(),
            })

            # Make a query dataframe by merging the graph_df and multimodal_df
            query_df = graph_df.merge(multimodal_df, how='cross')
            query_df = query_df[
                query_df.apply(
                    lambda x:
                    (x['q_node_name'].lower() in x['node_name'].lower()) & # node name
                    (x['node_type'] == x['q_node_type']), # node type
                    axis=1
                )
            ]
            query_df = query_df[['node_id', 'node_type', 'x', 'desc_x']].reset_index(drop=True)
            query_df['use_description'] = False # set to False for modal-specific embeddings

            # Update the state by adding the the selected node IDs
            state["selections"] = query_df.groupby("node_type")["node_id"].apply(list).to_dict()

        # Append a user prompt to the query dataframe
        query_df = pd.concat([
            query_df,
            pd.DataFrame({
                'node_id': 'user_prompt',
                'node_type': 'prompt',
                'x': prompt_emb,
                'desc_x': prompt_emb,
                'use_description': True # set to True for user prompt embedding
            })
        ]).reset_index(drop=True)

        return query_df

    def _perform_subgraph_extraction(self,
                                     state: Annotated[dict, InjectedState],
                                     cfg: dict,
                                     pyg_graph: Data,
                                     query_df: pd.DataFrame) -> dict:
        """
        Perform multimodal subgraph extraction based on modal-specific embeddings.

        Args:
            state: The injected state for the tool.
            cfg: The configuration dictionary.
            pyg_graph: The PyTorch Geometric graph Data.
            query_df: The DataFrame containing the query embeddings and modalities.

        Returns:
            A dictionary containing the extracted subgraph with nodes and edges.
        """
        # Initialize the subgraph dictionary
        subgraphs = {}
        subgraphs["nodes"] = []
        subgraphs["edges"] = []

        # Loop over query embeddings and modalities
        for q in query_df.iterrows():
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
            ).extract_subgraph(pyg_graph,
                               torch.tensor(q[1]['desc_x']), # description embedding
                               torch.tensor(q[1]['x']), # modal-specific embedding
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
            graph: The initial graph containing PyG and textualized graph.
            cfg: The configuration dictionary.

        Returns:
            A dictionary containing the PyG graph, NetworkX graph, and textualized graph.
        """
        # print(subgraph)
        # Prepare the PyTorch Geometric graph
        mapping = {n: i for i, n in enumerate(subgraph["nodes"].tolist())}
        pyg_graph = Data(
            # Node features
            # x=pyg_graph.x[subgraph["nodes"]],
            x=[graph["pyg"].x[i] for i in subgraph["nodes"]],
            node_id=np.array(graph["pyg"].node_id)[subgraph["nodes"]].tolist(),
            node_name=np.array(graph["pyg"].node_id)[subgraph["nodes"]].tolist(),
            enriched_node=np.array(graph["pyg"].enriched_node)[subgraph["nodes"]].tolist(),
            num_nodes=len(subgraph["nodes"]),
            # Edge features
            edge_index=torch.LongTensor(
                [
                    [
                        mapping[i]
                        for i in graph["pyg"].edge_index[:, subgraph["edges"]][0].tolist()
                    ],
                    [
                        mapping[i]
                        for i in graph["pyg"].edge_index[:, subgraph["edges"]][1].tolist()
                    ],
                ]
            ),
            edge_attr=graph["pyg"].edge_attr[subgraph["edges"]],
            edge_type=np.array(graph["pyg"].edge_type)[subgraph["edges"]].tolist(),
            relation=np.array(graph["pyg"].edge_type)[subgraph["edges"]].tolist(),
            label=np.array(graph["pyg"].edge_type)[subgraph["edges"]].tolist(),
            enriched_edge=np.array(graph["pyg"].enriched_edge)[subgraph["edges"]].tolist(),
        )

        # Networkx DiGraph construction to be visualized in the frontend
        nx_graph = nx.DiGraph()
        # Add nodes with attributes
        node_colors = {n: cfg.node_colors_dict[k]
                       for k, v in state["selections"].items() for n in v}
        for n in pyg_graph.node_name:
            nx_graph.add_node(n, color=node_colors.get(n, None))

        # Add edges with attributes
        edges = zip(
            pyg_graph.edge_index[0].tolist(),
            pyg_graph.edge_index[1].tolist(),
            pyg_graph.edge_type
        )
        for src, dst, edge_type in edges:
            nx_graph.add_edge(
                pyg_graph.node_name[src],
                pyg_graph.node_name[dst],
                relation=edge_type,
                label=edge_type,
            )

        # Prepare the textualized subgraph
        textualized_graph = (
            graph["text"]["nodes"].iloc[subgraph["nodes"]].to_csv(index=False)
            + "\n"
            + graph["text"]["edges"].iloc[subgraph["edges"]].to_csv(index=False)
        )

        return {
            "graph_pyg": pyg_graph,
            "graph_nx": nx_graph,
            "graph_text": textualized_graph,
        }

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
        with open(initial_graph["source"]["kg_pyg_path"], "rb") as f:
            initial_graph["pyg"] = pickle.load(f)
        with open(initial_graph["source"]["kg_text_path"], "rb") as f:
            initial_graph["text"] = pickle.load(f)

        # Prepare the query embeddings and modalities
        query_df = self._prepare_query_modalities(
            [EmbeddingWithOllama(model_name=cfg.ollama_embeddings[0]).embed_query(prompt)],
            state,
            initial_graph["pyg"]
        )

        # Perform subgraph extraction
        subgraphs = self._perform_subgraph_extraction(state,
                                                      cfg,
                                                      initial_graph["pyg"],
                                                      query_df)

        # Prepare subgraph as a NetworkX graph and textualized graph
        final_subgraph = self._prepare_final_subgraph(state,
                                                      subgraphs,
                                                      initial_graph,
                                                      cfg)

        # Prepare the dictionary of extracted graph
        dic_extracted_graph = {
            "name": arg_data.extraction_name,
            "tool_call_id": tool_call_id,
            "graph_source": initial_graph["source"]["name"],
            "topk_nodes": state["topk_nodes"],
            "topk_edges": state["topk_edges"],
            "graph_dict": {
                "nodes": list(final_subgraph["graph_nx"].nodes(data=True)),
                "edges": list(final_subgraph["graph_nx"].edges(data=True)),
            },
            "graph_text": final_subgraph["graph_text"],
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
