"""
Exctraction of multimodal subgraph using Prize-Collecting Steiner Tree (PCST) algorithm.
"""

from typing import Tuple, NamedTuple
import numpy as np
import cudf
import cuvs
import cupy as cp
import torch
import pcst_fast
from torch_geometric.data.data import Data

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

    def _compute_sim_scores(self, features_a, features_b, metric="cosine"):
        scores = cuvs.distance.pairwise_distance(features_a, features_b, metric=metric)
        scores = 1 - cp.asarray(scores).ravel()
        return scores

    def _compute_node_prizes(self,
                             graph_nodes: cudf.DataFrame,
                             query_emb: torch.Tensor,
                             modality: str) :
        """
        Compute the node prizes based on the cosine similarity between the query and nodes.

        Args:
            graph_nodes: The nodes dataframe of the graph.
            query_emb: The query embedding in PyTorch Tensor format. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            modality: The modality to use for the subgraph extraction based on the node type.

        Returns:
            The prizes of the nodes.
        """
        # Initialize variables
        sim_scores = cudf.Series(cp.zeros(len(graph_nodes), dtype=cp.float32))
        mask = (graph_nodes.node_type == modality)

        # Calculate cosine similarity for text features and update the score
        if self.use_description:
            sim_scores = _compute_sim_scores(
                graph_nodes["desc_x"].list.leaves.to_cupy().reshape(
                    -1, len(graph_nodes["desc_x"][0])
                ).astype(cp.float32),
                query_emb
            )  # shape [N, 1]
        else:
            sim_scores[mask] = _compute_sim_scores(
                graph_nodes[mask]["x"].list.leaves.to_cupy().reshape(
                    -1, len(graph_nodes[mask]["x"][0])
                ).astype(cp.float32),
                query_emb
            )  # shape [N, 1]

        # Set the prizes for nodes based on the similarity scores
        n_prizes = torch.tensor(graph_df.score.values, dtype=torch.float32)
        topk = min(self.topk, sim_scores.size)
        n_prizes = cudf.Series(0.0, index=cp.arange(sim_scores.size))
        n_prizes[(-sim_scores).sort_values()[:topk].index] = cp.arange(topk, 0, -1).astype(cp.float32)

        return n_prizes.to_cupy()

    def _compute_edge_prizes(self,
                             graph: Data,
                             text_emb: torch.Tensor) :
        """
        Compute the node prizes based on the cosine similarity between the query and nodes.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
            text_emb: The textual description embedding in PyTorch Tensor format.

        Returns:
            The prizes of the nodes.
        """
        # Note that as of now, the edge features are based on textual features
        # Compute prizes for edges
        e_prizes = _compute_sim_scores(
            graph_edges["edge_attr"].list.leaves.to_cupy().reshape(
                -1, len(graph_edges["edge_attr"][0])
            ).astype(cp.float32),
            text_emb
        )
        unique_prizes, inverse_indices = cp.unique(e_prizes, return_inverse=True)
        topk_e = min(topk_e, sim_scores.size) 
        topk_e_values = unique_prizes[cp.argsort(-unique_prizes)[:topk_e]]
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = inverse_indices == (unique_prizes == topk_e_values[k]).nonzero()[0]
            value = min((topk_e - k) / indices.sum().item(), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c_const)

        return e_prizes

    def compute_prizes(self,
                       graph: Data,
                       text_emb: torch.Tensor,
                       query_emb: torch.Tensor,
                       modality: str):
        """
        Compute the node prizes based on the cosine similarity between the query and nodes,
        as well as the edge prizes based on the cosine similarity between the query and edges.
        Note that the node and edge embeddings shall use the same embedding model and dimensions
        with the query.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
            text_emb: The textual description embedding in PyTorch Tensor format.
            query_emb: The query embedding in PyTorch Tensor format. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            modality: The modality to use for the subgraph extraction based on node type.

        Returns:
            The prizes of the nodes and edges.
        """
        # Compute prizes for nodes
        n_prizes = self._compute_node_prizes(graph, query_emb, modality)

        # Compute prizes for edges
        e_prizes = self._compute_edge_prizes(graph, text_emb)

        return {"nodes": n_prizes, "edges": e_prizes}

    def _create_edge_index(self, graph_nodes, graph_edges):
        # Create and additional node_index column
        graph_nodes = graph_nodes.reset_index(drop=True)
        graph_nodes['node_index'] = graph_nodes.index

        # Get head_index and tail_index
        edges = graph_edges.merge(graph_nodes[['node_id', 'node_index']],
                                left_on='head_id', right_on='node_id',
                                how='left').rename(columns={'node_index': 'head_index'}).drop(columns=['node_id'])
        edges = edges.merge(graph_nodes[['node_id', 'node_index']],
                            left_on='tail_id', right_on='node_id',
                            how='left').rename(columns={'node_index': 'tail_index'}).drop(columns=['node_id'])

        # Stacking to get into edge_index
        edge_index = cp.stack([
            edges['head_index'].to_cupy(),
            edges['tail_index'].to_cupy()
        ])

        return edge_index

    def compute_subgraph_costs(self,
                               graph_nodes,
                               graph_edges,
                               prizes: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the costs in constructing the subgraph proposed by G-Retriever paper.

        Args:
            graph_nodes: -
            graph_edges: -
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
        for i, (src, dst) in enumerate(self._create_edge_index(graph_nodes, graph_edges).T):
            prize_e = prizes["edges"][i]
            if prize_e <= updated_cost_e:
                mapping["edges"][len(edges)] = i
                edges.append((src, dst))
                costs.append(updated_cost_e - prize_e)
            else:
                virtual_node_id = graph.num_nodes + len(virtual["n_prizes"])
                mapping["nodes"][virtual_node_id] = i
                virtual["edges"].append((src, virtual_node_id))
                virtual["edges"].append((virtual_node_id, dst))
                virtual["costs"].append(0)
                virtual["costs"].append(0)
                virtual["n_prizes"].append(prize_e - updated_cost_e)
        prizes = np.concatenate([prizes["nodes"], np.array(virtual["n_prizes"])])
        edges_dict = {}
        edges_dict["edges"] = edges
        edges_dict["num_prior_edges"] = len(edges)
        # Final computation of the costs and edges based on the virtual costs and virtual edges
        if len(virtual["costs"]) > 0:
            costs = np.array(costs + virtual["costs"])
            edges = np.array(edges + virtual["edges"])
            edges_dict["edges"] = edges

        return edges_dict, prizes, costs, mapping

    def get_subgraph_nodes_edges(
        self, graph: Data, vertices: np.ndarray, edges_dict: dict, mapping: dict,
    ) -> dict:
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
        # Retrieve the selected nodes and edges based on the given vertices and edges
        subgraph_nodes = vertices[vertices < graph.num_nodes]
        subgraph_edges = [mapping["edges"][e] for e in edges if e < num_prior_edges]
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= graph.num_nodes]
            virtual_edges = [mapping["nodes"][i] for i in virtual_vertices]
            subgraph_edges = np.array(subgraph_edges + virtual_edges)
        edge_index = graph.edge_index[:, subgraph_edges]
        subgraph_nodes = np.unique(
            np.concatenate(
                [subgraph_nodes, edge_index[0].numpy(), edge_index[1].numpy()]
            )
        )

        return {"nodes": subgraph_nodes, "edges": subgraph_edges}

    def extract_subgraph(self,
                         graph: Data,
                         text_emb: torch.Tensor,
                         query_emb: torch.Tensor,
                         modality: str) -> dict:
        """
        Perform the Prize-Collecting Steiner Tree (PCST) algorithm to extract the subgraph.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
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
        prizes = self.compute_prizes(graph, text_emb, query_emb, modality)

        # Compute costs in constructing the subgraph
        edges_dict, prizes, costs, mapping = self.compute_subgraph_costs(
            graph, prizes
        )

        # Retrieve the subgraph using the PCST algorithm
        result_vertices, result_edges = pcst_fast.pcst_fast(
            edges_dict["edges"],
            prizes,
            costs,
            self.root,
            self.num_clusters,
            self.pruning,
            self.verbosity_level,
        )

        subgraph = self.get_subgraph_nodes_edges(
            graph,
            result_vertices,
            {"edges": result_edges, "num_prior_edges": edges_dict["num_prior_edges"]},
            mapping)

        return subgraph
