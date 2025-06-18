"""
Exctraction of multimodal subgraph using Prize-Collecting Steiner Tree (PCST) algorithm.
"""

from typing import Tuple, NamedTuple
import logging
import numpy as np
import cudf
from cuvs.distance import pairwise_distance
import cupy as cp
import torch
import pcst_fast
from torch_geometric.data.data import Data

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalPCSTPruning(NamedTuple):
    """
    Prize-Collecting Steiner Tree (PCST) pruning algorithm implementation inspired by G-Retriever
    (He et al., 'G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and
    Question Answering', NeurIPS 2024) paper.
    https://arxiv.org/abs/2402.07630
    https://github.com/XiaoxinHe/G-Retriever/blob/main/src/dataset/utils/retrieval.py

    Args:
        topk: The number of top nodes to consider.
        topk_e: The number of top edges to consider.
        cost_e: The cost of the edges.
        c_const: The constant value for the cost of the edges computation.
        root: The root node of the subgraph, -1 for unrooted.
        num_clusters: The number of clusters.
        pruning: The pruning strategy to use.
        verbosity_level: The verbosity level.
    """
    topk: int = 3
    topk_e: int = 3
    cost_e: float = 0.5
    c_const: float = 0.01
    root: int = -1
    num_clusters: int = 1
    pruning: str = "gw"
    verbosity_level: int = 0
    use_description: bool = False

    def _compute_sim_scores(self,
                            features_a: cp.ndarray,
                            features_b:  cp.ndarray,
                            metric: str="cosine"):
        """
        Compute the similarity scores between two sets of features using the specified metric.

        Args:
            features_a: The first set of features.
            features_b: The second set of features.
            metric: The metric to use for computing the similarity scores.

        Returns:
            The similarity scores between the two sets of features.
        """
        scores = pairwise_distance(features_a, features_b, metric=metric)
        scores = 1 - cp.asarray(scores).ravel()
        return scores

    def _compute_node_prizes(self,
                             graph_nodes: list,
                             query_emb: cp.ndarray,
                             modality: str) :
        """
        Compute the node prizes based on the cosine similarity between the query and nodes.

        Args:
            graph_nodes: The list
            query_emb: The query embedding in cupy array format. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            modality: The modality to use for the subgraph extraction based on the node type.

        Returns:
            The prizes of the nodes.
        """
        # Initialize variables
        sim_scores = cudf.Series(cp.zeros(len(graph_nodes['embedding']), dtype=cp.float32))

        # Calculate cosine similarity for text features and update the score
        logger.log(logging.INFO, "Computing node similarity scores")
        if self.use_description:
            sim_scores[:] = self._compute_sim_scores(
                graph_nodes['embedding']['desc_emb'].list.leaves.to_cupy().reshape(
                    graph_nodes['embedding'].shape[0], -1
                ).astype(cp.float32),
                query_emb
            )
        else:
            mask = graph_nodes['enrichment'].node_type == modality
            logger.log(logging.INFO, "Computing node similarity scores for modality: %s", modality)
            logger.log(logging.INFO, "Number of nodes with modality %s: %d", modality, mask.sum())
            logger.log(logging.INFO, "Length of graph nodes: %d", len(graph_nodes['embedding'][mask]))
            logger.log(logging.INFO, "Shape of embedding: %s", graph_nodes['embedding'][mask].shape)
            sim_scores[mask] = self._compute_sim_scores(
                graph_nodes['embedding'][mask]["feat_emb"].list.leaves.to_cupy().reshape(
                    graph_nodes['embedding'][mask].shape[0], -1
                ).astype(cp.float32),
                query_emb
            )

        # Set the prizes for nodes based on the similarity scores
        logger.log(logging.INFO, "Computing node prizes")
        topk = min(self.topk, sim_scores.size)
        n_prizes = cudf.Series(0.0, index=cp.arange(sim_scores.size))
        n_prizes[(-sim_scores).sort_values()[:topk].index] = cp.arange(topk,
                                                                       0, -1).astype(cp.float32)

        return n_prizes.to_cupy()

    def _compute_edge_prizes(self,
                             graph_edges: list,
                             text_emb: cp.ndarray) :
        """
        Compute the node prizes based on the cosine similarity between the query and nodes.

        Args:
            graph_edges: A list of edges dataframes of the graph.
            text_emb: The textual description embedding in cupy array format.

        Returns:
            The prizes of the nodes.
        """
        # Note that as of now, the edge features are based on textual features
        # Compute prizes for edges
        sim_dict = {}
        sim_dict['triplet_index'] = []
        sim_dict['score'] = []
        # Loop through the edges and compute the similarity scores
        for i, edges_emb_df in enumerate(graph_edges['embedding']):
            print(f"Processing edges chunk {i+1}/{len(graph_edges['embedding'])}")
            # Append the triplet indices
            sim_dict['triplet_index'].append(edges_emb_df['triplet_index'].to_cupy())

            # Calculate the similarity score
            sim_dict['score'].append(
                self._compute_sim_scores(edges_emb_df['edge_emb'].list.leaves.to_cupy()\
                    .reshape(edges_emb_df.shape[0], -1).astype(cp.float32),
                    text_emb)
                )

        # Create a DataFrame with the results
        e_prizes = cudf.DataFrame({
            'triplet_index': cp.concatenate(sim_dict['triplet_index']),
            'score': cp.concatenate(sim_dict['score'])
        })
        e_prizes = e_prizes.sort_values("triplet_index").reset_index(drop=True)
        e_prizes = e_prizes['score'].to_cupy()
        unique_prizes, inverse_indices = cp.unique(e_prizes, return_inverse=True)
        topk_e = min(self.topk_e, len(graph_edges))
        topk_e_values = unique_prizes[cp.argsort(-unique_prizes)[:topk_e]]
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = inverse_indices == (unique_prizes == topk_e_values[k]).nonzero()[0]
            value = min((topk_e - k) / indices.sum().item(), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - self.c_const)

        return e_prizes

    def compute_prizes(self,
                       graph: dict,
                       text_emb: torch.Tensor,
                       query_emb: torch.Tensor,
                       modality: str):
        """
        Compute the node prizes based on the cosine similarity between the query and nodes,
        as well as the edge prizes based on the cosine similarity between the query and edges.
        Note that the node and edge embeddings shall use the same embedding model and dimensions
        with the query.

        Args:
            graph: The graph data in a dictionary format, containing the nodes and edges dataframes.
            text_emb: The textual description embedding in PyTorch Tensor format.
            query_emb: The query embedding in PyTorch Tensor format. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            modality: The modality to use for the subgraph extraction based on node type.

        Returns:
            The prizes of the nodes and edges.
        """
        # Compute prizes for nodes
        logger.log(logging.INFO, "_compute_node_prizes")
        n_prizes = self._compute_node_prizes(graph["nodes"], query_emb, modality)

        # Compute prizes for edges
        logger.log(logging.INFO, "_compute_edge_prizes")
        e_prizes = self._compute_edge_prizes(graph["edges"], text_emb)

        return {"nodes": n_prizes, "edges": e_prizes}

    # def _create_edge_index(self,
    #                        graph_nodes: cudf.DataFrame,
    #                        graph_edges: cudf.DataFrame) -> cp.ndarray:
    #     """
    #     Create the edge index for the graph.

    #     Args:
    #         graph_nodes: The nodes dataframe of the graph.
    #         graph_edges: The edges dataframe of the graph.

    #     Returns:
    #         The edge index of the graph.
    #     """
    #     # Create and additional node_index column
    #     graph_nodes = graph_nodes.reset_index(drop=True)
    #     graph_nodes['node_index'] = graph_nodes.index

    #     # Get head_index and tail_index
    #     edges = graph_edges.merge(
    #         graph_nodes[['node_id', 'node_index']],
    #         left_on='head_id',
    #         right_on='node_id',
    #         how='left').rename(columns={'node_index': 'head_index'}).drop(columns=['node_id'])
    #     edges = edges.merge(
    #         graph_nodes[['node_id', 'node_index']],
    #         left_on='tail_id',
    #         right_on='node_id',
    #         how='left').rename(columns={'node_index': 'tail_index'}).drop(columns=['node_id'])

    #     # Stacking to get into edge_index
    #     edge_index = cp.stack([
    #         edges['head_index'].to_cupy(),
    #         edges['tail_index'].to_cupy()
    #     ])

    #     return edge_index

    def compute_subgraph_costs(self,
                               edge_index: cp.ndarray,
                               num_nodes: int,
                               prizes: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the costs in constructing the subgraph proposed by G-Retriever paper.

        Args:
            edge_index: The edge index of the graph.
            num_nodes: The number of nodes in the graph.
            prizes: The prizes of the nodes and the edges.

        Returns:
            edges: The edges of the subgraph, consisting of edges and number of edges without
                virtual edges.
            prizes: The prizes of the subgraph.
            costs: The costs of the subgraph.
        """
        # Logic to reduce the cost of the edges such that at least one edge is selected
        updated_cost_e = min(
            self.cost_e,
            prizes["edges"].max().item() * (1 - self.c_const / 2),
        )

        # Initialize variables
        edges = []
        costs = []
        virtual = {
            "n_prizes": [],
            "edges": [],
            "costs": [],
        }
        mapping = {"nodes": {}, "edges": {}}

        # Compute the costs, edges, and virtual variables based on the prizes
        for i, (src, dst) in enumerate(edge_index.T):
            prize_e = prizes["edges"][i].item()
            if prize_e <= updated_cost_e:
                mapping["edges"][len(edges)] = i
                edges.append((src.item(), dst.item()))
                costs.append(updated_cost_e - prize_e)
            else:
                virtual_node_id = num_nodes + len(virtual["n_prizes"])
                mapping["nodes"][virtual_node_id] = i
                virtual["edges"].append((src.item(), virtual_node_id))
                virtual["edges"].append((virtual_node_id, dst.item()))
                virtual["costs"].append(0)
                virtual["costs"].append(0)
                virtual["n_prizes"].append(prize_e - updated_cost_e)
        prizes = cp.concatenate([prizes["nodes"], cp.array(virtual["n_prizes"])])
        edges_dict = {}
        edges_dict["edges"] = edges
        edges_dict["num_prior_edges"] = len(edges)
        # Final computation of the costs and edges based on the virtual costs and virtual edges
        if len(virtual["costs"]) > 0:
            costs = cp.array(costs + virtual["costs"])
            edges = cp.array(edges + virtual["edges"])
            edges_dict["edges"] = edges

        return edges_dict, prizes, costs, mapping


    def get_subgraph_nodes_edges(self,
                                 num_nodes: int,
                                 vertices: cp.ndarray,
                                 edges_dict: dict,
                                 mapping: dict) -> dict:
        """
        Get the selected nodes and edges of the subgraph based on the vertices and edges computed
        by the PCST algorithm.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
            vertices: The vertices of the subgraph computed by the PCST algorithm.
            edges_dict: The dictionary of edges of the subgraph computed by the PCST algorithm,
                and the number of prior edges (without virtual edges).
            mapping: The mapping dictionary of the nodes and edges.
            num_prior_edges: The number of edges before adding virtual edges.

        Returns:
            The selected nodes and edges of the extracted subgraph.
        """
        # Get edges information
        edges = edges_dict["edges"]
        num_prior_edges = edges_dict["num_prior_edges"]
        # Get edges information
        edges = edges_dict["edges"]
        num_prior_edges = edges_dict["num_prior_edges"]
        # Retrieve the selected nodes and edges based on the given vertices and edges
        subgraph_nodes = vertices[vertices < num_nodes]
        subgraph_edges = [mapping["edges"][e.item()] for e in edges if e < num_prior_edges]
        virtual_vertices = vertices[vertices >= num_nodes]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= num_nodes]
            virtual_edges = [mapping["nodes"][i.item()] for i in virtual_vertices]
            subgraph_edges = cp.array(subgraph_edges + virtual_edges)
        edge_index = edges_dict["edge_index"][:, subgraph_edges]
        subgraph_nodes = cp.unique(
            cp.concatenate(
                [subgraph_nodes, edge_index[0], edge_index[1]]
            )
        )

        return {"nodes": subgraph_nodes, "edges": subgraph_edges}

    def extract_subgraph(self,
                         graph: dict,
                         text_emb: torch.Tensor,
                         query_emb: torch.Tensor,
                         modality: str) -> dict:
        """
        Perform the Prize-Collecting Steiner Tree (PCST) algorithm to extract the subgraph.

        Args:
            graph: The graph dictionary.
            text_emb: The textual description embedding in PyTorch Tensor format.
            query_emb: The query embedding in PyTorch Tensor format. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            modality: The modality to use for the subgraph extraction
                (e.g., "text", "sequence", "smiles").

        Returns:
            The selected nodes and edges of the subgraph.
        """
        # Assert the topk and topk_e values for subgraph retrieval
        assert self.topk > 0, "topk must be greater than or equal to 0"
        assert self.topk_e > 0, "topk_e must be greater than or equal to 0"

        # Retrieve the top-k nodes and edges based on the query embedding
        logger.log(logging.INFO, "compute_prizes")
        prizes = self.compute_prizes(graph, text_emb, query_emb, modality)

        # Create the edge index for the graph
        logger.log(logging.INFO, "Creating edge index")
        edges_df_sorted = graph["edges"]["enrichment"].sort_values("triplet_index",
                                                                   ignore_index=True)
        edge_index = cp.stack([
            edges_df_sorted["head_index"].to_cupy(),
            edges_df_sorted["tail_index"].to_cupy()
        ])

        # Compute costs in constructing the subgraph
        logger.log(logging.INFO, "compute_subgraph_costs")
        edges_dict, prizes, costs, mapping = self.compute_subgraph_costs(
            edge_index, len(graph["nodes"]["enrichment"]), prizes
        )

        # Retrieve the subgraph using the PCST algorithm
        logger.log(logging.INFO, "Running PCST algorithm")
        result_vertices, result_edges = pcst_fast.pcst_fast(
            edges_dict["edges"].get(),
            prizes.get(),
            costs.get(),
            self.root,
            self.num_clusters,
            self.pruning,
            self.verbosity_level,
        )

        # Get subgraph nodes and edges based on the result of the PCST algorithm
        logger.log(logging.INFO, "Getting subgraph nodes and edges")
        subgraph = self.get_subgraph_nodes_edges(
            len(graph["nodes"]["enrichment"]),
            cp.asarray(result_vertices),
            {"edges": cp.asarray(result_edges),
             "num_prior_edges": edges_dict["num_prior_edges"],
             "edge_index": edge_index},
            mapping)

        return subgraph
