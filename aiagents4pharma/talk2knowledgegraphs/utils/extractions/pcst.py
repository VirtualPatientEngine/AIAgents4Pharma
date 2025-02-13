"""
Exctraction of subgraph using Prize-Collecting Steiner Tree (PCST) algorithm.
"""

from typing import Tuple, NamedTuple
import numpy as np
import torch
import pcst_fast
from torch_geometric.data.data import Data

class PCSTPruning(NamedTuple):
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

    def compute_prizes(self, graph: Data, query_emb: torch.Tensor) -> np.ndarray:
        """
        Compute the node prizes based on the cosine similarity between the query and nodes,
        as well as the edge prizes based on the cosine similarity between the query and edges.
        Note that the node and edge embeddings shall use the same embedding model and dimensions
        with the query.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
            query_emb: The query embedding in PyTorch Tensor format.

        Returns:
            The prizes of the nodes and edges.
        """
        # Compute prizes for nodes
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(query_emb, graph.x)
        topk = min(self.topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)
        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()

        # Compute prizes for edges
        # e_prizes = torch.nn.CosineSimilarity(dim=-1)(query_emb, graph.edge_attr)
        # topk_e = min(self.topk_e, e_prizes.unique().size(0))
        # topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        # e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        # last_topk_e_value = topk_e
        # for k in range(topk_e):
        #     indices = e_prizes == topk_e_values[k]
        #     value = min((topk_e - k) / sum(indices), last_topk_e_value)
        #     e_prizes[indices] = value
        #     last_topk_e_value = value * (1 - self.c_const)

        # Optimized version of the above code
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(query_emb, graph.edge_attr)
        unique_prizes, inverse_indices = e_prizes.unique(return_inverse=True)
        topk_e = min(self.topk_e, unique_prizes.size(0))
        topk_e_values, _ = torch.topk(unique_prizes, topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = inverse_indices == (
                unique_prizes == topk_e_values[k]
            ).nonzero(as_tuple=True)[0]
            value = min((topk_e - k) / indices.sum().item(), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - self.c_const)

        return {"nodes": n_prizes, "edges": e_prizes}

    def compute_subgraph_costs(
        self, graph: Data, prizes: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the costs in constructing the subgraph proposed by G-Retriever paper.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
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
        for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
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

    def extract_subgraph(self, graph: Data, query_emb: torch.Tensor) -> dict:
        """
        Perform the Prize-Collecting Steiner Tree (PCST) algorithm to extract the subgraph.

        Args:
            graph: The knowledge graph in PyTorch Geometric Data format.
            query_emb: The query embedding.

        Returns:
            The selected nodes and edges of the subgraph.
        """
        # Assert the topk and topk_e values for subgraph retrieval
        assert self.topk > 0, "topk must be greater than or equal to 0"
        assert self.topk_e > 0, "topk_e must be greater than or equal to 0"

        # Retrieve the top-k nodes and edges based on the query embedding
        prizes = self.compute_prizes(graph, query_emb)

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
