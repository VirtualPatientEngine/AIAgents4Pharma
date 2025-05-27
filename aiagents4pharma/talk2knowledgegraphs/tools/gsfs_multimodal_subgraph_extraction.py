"""
Tool for performing multimodal subgraph extraction.
"""

import os
import glob
from typing import Type, Annotated
import logging
import numpy as np
import cudf
import cupy as cp
import hydra
# import networkx as nx
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
import torch
# from torch_geometric.data import Data

# import torch
# import cudf
from torch_geometric.data import TensorAttr
from cugraph_pyg.data import GraphStore, TensorDictFeatureStore

# from ..utils.extractions.cu_multimodal_pcst import MultimodalPCSTPruning
from ..utils.extractions.gsfs_multimodal_pcst import MultimodalPCSTPruning
# from ..utils.embeddings.ollama import EmbeddingWithOllama
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

    def _load_graph(self, cfg, state: Annotated[dict, InjectedState]) -> dict:
        """
        Load the graph from the specified path in the configuration.

        Args:
            cfg: The configuration dictionary.
            state: The injected state for the tool.

        Returns:
            A dictionary containing the graph with nodes and edges.
        """
        logger.log(logging.INFO, "Loading graph from %s", cfg.biobridge.source)
        # Retrieve source graph from the state
        graph = {}
        graph["source"] = state["dic_source_graph"][-1]  # The last source graph as of now
        # logger.log(logging.INFO, "Source graph: %s", source_graph)

        # Load the knowledge graph
        # initial_graph["nodes"] = cudf.read_parquet(initial_graph["source"]["kg_nodes_path"])
        # initial_graph["edges"] = cudf.read_parquet(initial_graph["source"]["kg_edges_path"])

        # Loop over nodes and edges
        for element in ["nodes", "edges"]:
            # Make an empty dictionary for each folder
            graph[element] = {}
            for stage in ["enrichment", "embedding"]:
                # print(element, stage)
                # Create the file pattern for the current subfolder
                file_list = glob.glob(os.path.join(cfg.biobridge.source,
                                                   element,
                                                   stage,
                                                   '*.parquet.gzip'))
                # logger.log(logging.INFO, "File list for %s/%s: %s", element, stage, file_list)
                # if element != "edges" and stage == "embedding":
                # Read and concatenate all dataframes in the folder
                graph[element][stage] = cudf.concat([
                    cudf.read_parquet(f) for f in file_list
                ], ignore_index=True)

        return graph

    def _construct_store(self, graph: dict) -> dict:
        """
        Construct the GraphStore and TensorDictFeatureStore from the graph data.
        
        Args:
            graph: The graph dictionary containing nodes and edges.
            
        Returns:
            A dictionary containing the GraphStore, TensorDictFeatureStore, and mapper.
        """
        logger.log(logging.INFO, "Construct GraphStore and TensorDictFeautureStore")

        # Initialize FeatureStore and mapper
        graph_store = GraphStore()
        feature_store = TensorDictFeatureStore()
        mapper = {}

        # Get nodes enrichment and embedding dataframes
        nodes_enrichment_df = graph['nodes']['enrichment']
        nodes_embedding_df = graph['nodes']['embedding']
        # Get edges enrichment and embedding dataframes
        edges_enrichment_df = graph['edges']['enrichment']
        edges_embedding_df = graph['edges']['embedding']

        # Loop over group enrichment nodes by type
        for nt, nodes_df in nodes_enrichment_df.groupby('node_type'):
            # print(f"Node type: {nt}")
            node_count = len(nodes_df)
            # print(f"Number of nodes: {node_count}")

            # Get node_ids
            emb_df = nodes_embedding_df[nodes_embedding_df['node_id'].isin(nodes_df['node_id'])]

            # Sort both by node_id for alignment
            nodes_df_sorted = nodes_df.sort_values('node_id')
            emb_df_sorted = emb_df.sort_values('node_id')

            # Ensure sorted node_ids match
            assert cudf.Series.equals(nodes_df_sorted['node_id'].reset_index(drop=True),
                                      emb_df_sorted['node_id'].reset_index(drop=True)), \
                                        f"Node ID mismatch in {nt} after sorting"

            # Get node_index as torch tensor directly
            node_index_tensor = torch.tensor(nodes_df_sorted["node_index"].to_numpy(),
                                             dtype=torch.int64)
            feature_store[TensorAttr(group_name=nt, attr_name="node_index")] = node_index_tensor

            # # Construct mapper for node_index
            node_index_list = nodes_df_sorted["node_index"].to_numpy().tolist()
            mapper[nt] = {
                'to_node_index': dict(enumerate(node_index_list)),
                'from_node_index': {v: i for i, v in enumerate(node_index_list)}
            }

            # Convert embeddings as tensors and add to FeatureStore
            for attr_name in ["desc_emb", "feat_emb"]:
                emb_tensor = torch.tensor(emb_df_sorted[attr_name].to_arrow().to_pylist(),
                                          dtype=torch.float32)
                feature_store[TensorAttr(group_name=nt, attr_name=attr_name)] = emb_tensor

        # Loop over edge types
        for edge_type_str in edges_enrichment_df['edge_type_str'].unique().to_arrow().to_pylist():
            # print(f"Processing edge type: {edge_type_str}")
            src_type, rel_type, tgt_type = edge_type_str.split('|')

            # Filter edges for this edge_type_str once
            filtered_df = edges_enrichment_df[
                edges_enrichment_df['edge_type_str'] == edge_type_str
            ][['triplet_index', 'head_index', 'tail_index']]

            # Convert mapper dicts to cudf Series for vectorized mapping
            src_map = cudf.Series(mapper[src_type]['from_node_index'])
            tgt_map = cudf.Series(mapper[tgt_type]['from_node_index'])

            # Vectorized mapping of head_index and tail_index using replace (works like dict lookup)
            mapped_head = filtered_df['head_index'].replace(src_map).astype('int64')
            mapped_tail = filtered_df['tail_index'].replace(tgt_map).astype('int64')

            # Check if mapping was successful
            if mapped_head.isnull().any() or mapped_tail.isnull().any():
                raise ValueError(f"Mapping failure for edge type {edge_type_str}")

            # Edge index
            edge_index = torch.tensor(
                cudf.concat([mapped_head, mapped_tail], axis=1).to_pandas().values.T,
                dtype=torch.long
            ).contiguous()

            # Store edge index in the GraphStore
            graph_store[(src_type, rel_type, tgt_type), "coo"] = edge_index

            # Add triplet index to the FeatureStore
            triplet_index_tensor = torch.tensor(filtered_df['triplet_index'].to_numpy(),
                                                dtype=torch.long).unsqueeze(0)
            feature_store[TensorAttr(group_name=(src_type, rel_type, tgt_type),
                                     attr_name='triplet_index')] = triplet_index_tensor

            # Store edge embeddings in the FeatureStore
            edge_emb_df = edges_embedding_df[edges_embedding_df['edge_type_str'] == edge_type_str]

            # Convert edge embeddings to torch tensor
            edge_emb_tensor = torch.tensor(edge_emb_df['edge_emb'].to_arrow().to_pylist(),
                                           dtype=torch.float32).unsqueeze(0)
            feature_store[TensorAttr(group_name=(src_type, rel_type, tgt_type),
                                     attr_name='edge_emb')] = edge_emb_tensor

        return {"graph_store": graph_store,
                "feature_store": feature_store,
                "mapper": mapper}

    def _prepare_query_modalities(self,
                                  prompt: str,
                                  state: Annotated[dict, InjectedState],
                                  graph: dict) -> cudf.DataFrame:
        """
        Prepare the modality-specific query for subgraph extraction.

        Args:
            prompt_emb: The embedding of the user prompt in a list.
            state: The injected state for the tool.
            graph: The graph dictionary containing nodes and edges.

        Returns:
            A DataFrame containing the query embeddings and modalities.
        """
        # Initialize dataframes
        multimodal_df = cudf.DataFrame({"name": [], "node_type": []})
        query_df = cudf.DataFrame({"node_id": [],
                                   "node_type": [],
                                   "feat": [],
                                   "desc": [],
                                   "feat_emb": [],
                                   "desc_emb": [],
                                   "use_description": []})

        # Loop over the uploaded files and find multimodal files
        for i in range(len(state["uploaded_files"])):
            # Check if multimodal file is uploaded
            if state["uploaded_files"][i]["file_type"] == "multimodal":
                # Read the csv file
                multimodal_df = cudf.read_csv(state["uploaded_files"][i]["file_path"])

        # Check if the multimodal_df is empty
        if len(multimodal_df) > 0:
            # Prepare multimodal_df
            multimodal_df.rename(columns={"name": "q_node_name",
                                            "node_type": "q_node_type"}, inplace=True)

            # Make and process a query dataframe by merging the graph_df and multimodal_df
            query_df = graph["nodes"]["enrichment"][
                ['node_id', 'node_name', 'node_type', 'desc', 'feat']
            ].merge(multimodal_df, how='cross')
            query_df['q_node_name'] = query_df['q_node_name'].str.lower()
            query_df['node_name'] = query_df['node_name'].str.lower()
            # Get the mask for filtering based on the query
            mask = (
                query_df['node_name'].str.contains(query_df['q_node_name']) &
                (query_df['node_type'] == query_df['q_node_type'])
            )
            query_df = query_df[mask]
            query_df = query_df[['node_id',
                                 'node_type',
                                 'feat',
                                 'desc']].reset_index(drop=True)
            query_df['use_description'] = False # set to False for modal-specific embeddings

            # Merge the query dataframe with the node embeddings
            # Re-arrange the columns to include embeddings
            query_df = query_df.merge(
                graph["nodes"]["embedding"][["node_id", "desc_emb", "feat_emb"]],
                on="node_id",
                how="left"
            )[['node_id', 'node_type', 'feat', 'desc', 'desc_emb', 'feat_emb', 'use_description']]

            # Update the state by adding the the selected node IDs
            state["selections"] = query_df.to_pandas().groupby(
                "node_type"
            )["node_id"].apply(list).to_dict()

        # Append a user prompt to the query dataframe
        prompt_emb = [state["embedding_model"].embed_query(prompt)]
        query_df = cudf.concat([
            query_df,
            cudf.DataFrame({
                'node_id': 'user_prompt',
                'node_type': 'prompt',
                'feat': prompt,
                'desc': prompt,
                'feat_emb': prompt_emb,
                'desc_emb': prompt_emb,
                'use_description': True # set to True for user prompt embedding
            })
        ]).reset_index(drop=True)

        # logger.log(logging.INFO, "Query DataFrame prepared with %d rows", len(query_df))
        # logger.log(logging.INFO, "Query DataFrame:\n%s", query_df.node_id.to_arrow().to_pylist())
        # logger.log(logging.INFO, "Prompt embedding: %s", prompt_emb)

        return query_df

    def _perform_subgraph_extraction(self,
                                     query_df: cudf.DataFrame,
                                     state: Annotated[dict, InjectedState],
                                     graph: dict,
                                     store: dict,
                                     cost_e: float = 0.5,
                                     c_const: float = 0.01,
                                     root: int = -1,
                                     num_clusters: int = 1,
                                     pruning: str = "gw",
                                     verbosity_level: int = 0,) -> dict:
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
            logger.log(logging.INFO, "Processing query %s", {q[1]['node_id']})
            # Prepare the PCSTPruning object and extract the subgraph
            # Parameters were set in the configuration file obtained from Hydra
            subgraph = MultimodalPCSTPruning(
                topk=state["topk_nodes"],
                topk_e=state["topk_edges"],
                cost_e=cost_e,
                c_const=c_const,
                root=root,
                num_clusters=num_clusters,
                pruning=pruning,
                verbosity_level=verbosity_level,
                use_description=q[1]['use_description'],
            ).extract_subgraph(graph,
                               store,
                               {"text_emb": cp.array(
                                   q[1]['desc_emb'], dtype=cp.float32
                                ).reshape(1, -1),
                                "emb": cp.array(
                                    q[1]['feat_emb'], dtype=cp.float32
                                ).reshape(1, -1).astype(cp.float32),
                                "modality": q[1]['node_type']})

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
                                subgraph: dict,
                                cfg: dict,
                                state:Annotated[dict, InjectedState],
                                graph: dict) -> dict:
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
        # print(color_df)

        # Prepare graph dataframes
        # Nodes
        graph_nodes = graph["nodes"]["enrichment"].sort_values("node_index",
                                                               ignore_index=True).copy()
        graph_nodes = graph_nodes.iloc[subgraph['nodes']][
            ['node_id', 'node_name', 'node_type', 'desc', 'feat']
        ]
        if not color_df.empty:
            # Merge the color dataframe with the graph nodes
            graph_nodes = graph_nodes.merge(color_df, on="node_id", how="left")
        else:
            graph_nodes["color"] = 'black'  # Default color
        graph_nodes['color'].fillna('black', inplace=True) # Fill NaN colors with black
        # Edges
        graph_edges = graph["edges"]["enrichment"].sort_values("triplet_index",
                                                               ignore_index=True).copy()
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

        # Load graph dictionary
        logger.log(logging.INFO, "_load_graph")
        graph = self._load_graph(cfg, state)

        # Construct the GraphStore and TensorDictFeatureStore from the graph data
        logger.log(logging.INFO, "_construct_store")
        store = self._construct_store(graph)

        # Prepare the query embeddings and modalities
        logger.log(logging.INFO, "_prepare_query_modalities")
        query_df = self._prepare_query_modalities(prompt, state, graph)

        # # Perform subgraph extraction
        logger.log(logging.INFO, "_perform_subgraph_extraction")
        subgraphs = self._perform_subgraph_extraction(query_df, state, graph, store,
                                                      cost_e=cfg.cost_e,
                                                      c_const=cfg.c_const,
                                                      root=cfg.root,
                                                      num_clusters=cfg.num_clusters,
                                                      pruning=cfg.pruning,
                                                      verbosity_level=cfg.verbosity_level,)

        # # Prepare subgraph as a NetworkX graph and textualized graph
        logger.log(logging.INFO, "_prepare_final_subgraph")
        final_subgraph = self._prepare_final_subgraph(subgraphs,
                                                      cfg,
                                                      state,
                                                      graph,)

        # # Prepare the dictionary of extracted graph
        logger.log(logging.INFO, "dic_extracted_graph")
        dic_extracted_graph = {
            "name": arg_data.extraction_name,
            "tool_call_id": tool_call_id,
            "graph_source": graph["source"]["name"],
            "topk_nodes": state["topk_nodes"],
            "topk_edges": state["topk_edges"],
            "graph_dict": {
                "nodes": final_subgraph["nodes"],
                "edges": final_subgraph["edges"],
            },
            "graph_text": final_subgraph["text"],
            "graph_summary": None,
        }

        # # Prepare the dictionary of updated state
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
