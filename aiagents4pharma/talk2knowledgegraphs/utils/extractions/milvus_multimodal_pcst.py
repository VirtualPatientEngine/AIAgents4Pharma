"""
Exctraction of multimodal subgraph using Prize-Collecting Steiner Tree (PCST) algorithm.
"""

import logging
import pickle
from typing import NamedTuple

import pandas as pd
import pcst_fast
from pymilvus import Collection

try:
    import cudf
    import cupy as py

    df = cudf
except ImportError:
    import numpy as py

    df = pd

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
    metric_type: str = "IP"  # Inner Product

    def prepare_collections(self, cfg: dict, modality: str) -> dict:
        """
        Prepare the collections for nodes, node-type specific nodes, and edges in Milvus.

        Args:
            cfg: The configuration dictionary containing the Milvus setup.
            modality: The modality to use for the subgraph extraction.

        Returns:
            A dictionary containing the collections of nodes, node-type specific nodes, and edges.
        """
        # Initialize the collections dictionary
        colls = {}

        # Load the collection for nodes
        colls["nodes"] = Collection(name=f"{cfg.milvus_db.database_name}_nodes")

        if modality != "prompt":
            # Load the collection for the specific node type
            colls["nodes_type"] = Collection(
                f"{cfg.milvus_db.database_name}_nodes_{modality.replace('/', '_')}"
            )

        # Load the collection for edges
        colls["edges"] = Collection(name=f"{cfg.milvus_db.database_name}_edges")

        # Load the collections
        for coll in colls.values():
            coll.load()

        return colls

    def _compute_node_prizes(self, query_emb: list, colls: dict) -> dict:
        """
        Compute the node prizes based on the cosine similarity between the query and nodes.

        Args:
            query_emb: The query embedding. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            colls: The collections of nodes, node-type specific nodes, and edges in Milvus.

        Returns:
            The prizes of the nodes.
        """
        # Intialize several variables
        topk = min(self.topk, colls["nodes"].num_entities)
        n_prizes = py.zeros(colls["nodes"].num_entities, dtype=py.float32)

        # Calculate cosine similarity for text features and update the score
        if self.use_description:
            # Search the collection with the text embedding
            res = colls["nodes"].search(
                data=[query_emb],
                anns_field="desc_emb",
                param={"metric_type": self.metric_type},
                limit=topk,
                output_fields=["node_id"],
            )
        else:
            # Search the collection with the query embedding
            res = colls["nodes_type"].search(
                data=[query_emb],
                anns_field="feat_emb",
                param={"metric_type": self.metric_type},
                limit=topk,
                output_fields=["node_id"],
            )

        # Update the prizes based on the search results
        n_prizes[[r.id for r in res[0]]] = py.arange(topk, 0, -1).astype(py.float32)

        return n_prizes

    def _compute_edge_prizes(self, text_emb: list, colls: dict) -> py.ndarray:
        """
        Compute the node prizes based on the cosine similarity between the query and nodes.

        Args:
            text_emb: The textual description embedding.
            colls: The collections of nodes, node-type specific nodes, and edges in Milvus.

        Returns:
            The prizes of the nodes.
        """
        # Intialize several variables
        topk_e = min(self.topk_e, colls["edges"].num_entities)
        e_prizes = py.zeros(colls["edges"].num_entities, dtype=py.float32)

        # Search the collection with the query embedding
        res = colls["edges"].search(
            data=[text_emb],
            anns_field="feat_emb",
            param={"metric_type": self.metric_type},
            limit=topk_e,  # Only retrieve the top-k edges
            # limit=colls["edges"].num_entities,
            output_fields=["head_id", "tail_id"],
        )

        # Update the prizes based on the search results
        e_prizes[[r.id for r in res[0]]] = [r.score for r in res[0]]

        # Further process the edge_prizes
        unique_prizes, inverse_indices = py.unique(e_prizes, return_inverse=True)
        topk_e_values = unique_prizes[py.argsort(-unique_prizes)[:topk_e]]
        # e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = inverse_indices == (unique_prizes == topk_e_values[k]).nonzero()[0]
            value = min((topk_e - k) / indices.sum().item(), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - self.c_const)

        return e_prizes

    def compute_prizes(self, text_emb: list, query_emb: list, colls: dict) -> dict:
        """
        Compute the node prizes based on the cosine similarity between the query and nodes,
        as well as the edge prizes based on the cosine similarity between the query and edges.
        Note that the node and edge embeddings shall use the same embedding model and dimensions
        with the query.

        Args:
            text_emb: The textual description embedding.
            query_emb: The query embedding. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            colls: The collections of nodes, node-type specific nodes, and edges in Milvus.

        Returns:
            The prizes of the nodes and edges.
        """
        # Compute prizes for nodes
        logger.log(logging.INFO, "_compute_node_prizes")
        n_prizes = self._compute_node_prizes(query_emb, colls)

        # Compute prizes for edges
        logger.log(logging.INFO, "_compute_edge_prizes")
        e_prizes = self._compute_edge_prizes(text_emb, colls)

        return {"nodes": n_prizes, "edges": e_prizes}

    def compute_subgraph_costs(
        self, edge_index: py.ndarray, num_nodes: int, prizes: dict
    ) -> tuple[py.ndarray, py.ndarray, py.ndarray]:
        """
        Compute the costs in constructing the subgraph proposed by G-Retriever paper.

        Args:
            edge_index: The edge index of the graph, consisting of source and destination nodes.
            num_nodes: The number of nodes in the graph.
            prizes: The prizes of the nodes and the edges.

        Returns:
            edges: The edges of the subgraph, consisting of edges and number of edges without
                virtual edges.
            prizes: The prizes of the subgraph.
            costs: The costs of the subgraph.
        """
        # Initialize several variables
        real_ = {}
        virt_ = {}

        # Update edge cost threshold
        updated_cost_e = min(
            self.cost_e,
            py.max(prizes["edges"]).item() * (1 - self.c_const / 2),
        )

        # Masks for real and virtual edges
        logger.log(logging.INFO, "Creating masks for real and virtual edges")
        real_["mask"] = prizes["edges"] <= updated_cost_e
        virt_["mask"] = ~real_["mask"]

        # Real edge indices
        logger.log(logging.INFO, "Computing real edges")
        real_["indices"] = py.nonzero(real_["mask"])[0]
        real_["src"] = edge_index[0][real_["indices"]]
        real_["dst"] = edge_index[1][real_["indices"]]
        real_["edges"] = py.stack([real_["src"], real_["dst"]], axis=1)
        real_["costs"] = updated_cost_e - prizes["edges"][real_["indices"]]

        # Edge index mapping: local real edge idx -> original global index
        logger.log(logging.INFO, "Creating mapping for real edges")
        mapping_edges = dict(
            zip(range(len(real_["indices"])), real_["indices"].tolist(), strict=False)
        )

        # Virtual edge handling
        logger.log(logging.INFO, "Computing virtual edges")
        virt_["indices"] = py.nonzero(virt_["mask"])[0]
        virt_["src"] = edge_index[0][virt_["indices"]]
        virt_["dst"] = edge_index[1][virt_["indices"]]
        virt_["prizes"] = prizes["edges"][virt_["indices"]] - updated_cost_e

        # Generate virtual node IDs
        logger.log(logging.INFO, "Generating virtual node IDs")
        virt_["num"] = virt_["indices"].shape[0]
        virt_["node_ids"] = py.arange(num_nodes, num_nodes + virt_["num"])

        # Virtual edges: (src → virtual), (virtual → dst)
        logger.log(logging.INFO, "Creating virtual edges")
        virt_["edges_1"] = py.stack([virt_["src"], virt_["node_ids"]], axis=1)
        virt_["edges_2"] = py.stack([virt_["node_ids"], virt_["dst"]], axis=1)
        virt_["edges"] = py.concatenate([virt_["edges_1"], virt_["edges_2"]], axis=0)
        virt_["costs"] = py.zeros((virt_["edges"].shape[0],), dtype=real_["costs"].dtype)

        # Combine real and virtual edges/costs
        logger.log(logging.INFO, "Combining real and virtual edges/costs")
        all_edges = py.concatenate([real_["edges"], virt_["edges"]], axis=0)
        all_costs = py.concatenate([real_["costs"], virt_["costs"]], axis=0)

        # Final prizes
        logger.log(logging.INFO, "Getting final prizes")
        final_prizes = py.concatenate([prizes["nodes"], virt_["prizes"]], axis=0)

        # Mapping virtual node ID -> edge index in original graph
        logger.log(logging.INFO, "Creating mapping for virtual nodes")
        mapping_nodes = dict(
            zip(virt_["node_ids"].tolist(), virt_["indices"].tolist(), strict=False)
        )

        # Build return values
        logger.log(logging.INFO, "Building return values")
        edges_dict = {
            "edges": all_edges,
            "num_prior_edges": real_["edges"].shape[0],
        }
        mapping = {
            "edges": mapping_edges,
            "nodes": mapping_nodes,
        }

        return edges_dict, final_prizes, all_costs, mapping

    def get_subgraph_nodes_edges(
        self, num_nodes: int, vertices: py.ndarray, edges_dict: dict, mapping: dict
    ) -> dict:
        """
        Get the selected nodes and edges of the subgraph based on the vertices and edges computed
        by the PCST algorithm.

        Args:
            num_nodes: The number of nodes in the graph.
            vertices: The vertices selected by the PCST algorithm.
            edges_dict: A dictionary containing the edges and the number of prior edges.
            mapping: A dictionary containing the mapping of nodes and edges.

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
            subgraph_edges = py.array(subgraph_edges + virtual_edges)
        edge_index = edges_dict["edge_index"][:, subgraph_edges]
        subgraph_nodes = py.unique(py.concatenate([subgraph_nodes, edge_index[0], edge_index[1]]))

        return {"nodes": subgraph_nodes, "edges": subgraph_edges}

    def extract_subgraph(self, text_emb: list, query_emb: list, modality: str, cfg: dict) -> dict:
        """
        Perform the Prize-Collecting Steiner Tree (PCST) algorithm to extract the subgraph.

        Args:
            text_emb: The textual description embedding.
            query_emb: The query embedding. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            modality: The modality to use for the subgraph extraction
                (e.g., "text", "sequence", "smiles").
            cfg: The configuration dictionary containing the Milvus setup.

        Returns:
            The selected nodes and edges of the subgraph.
        """
        # Load the collections for nodes
        logger.log(logging.INFO, "Preparing collections")
        colls = self.prepare_collections(cfg, modality)

        # Load cache edge index
        logger.log(logging.INFO, "Loading cache edge index")
        with open(cfg.milvus_db.cache_edge_index_path, "rb") as f:
            edge_index = pickle.load(f)
            edge_index = py.array(edge_index)

        # Assert the topk and topk_e values for subgraph retrieval
        assert self.topk > 0, "topk must be greater than or equal to 0"
        assert self.topk_e > 0, "topk_e must be greater than or equal to 0"

        # Retrieve the top-k nodes and edges based on the query embedding
        logger.log(logging.INFO, "compute_prizes")
        prizes = self.compute_prizes(text_emb, query_emb, colls)

        # Compute costs in constructing the subgraph
        logger.log(logging.INFO, "compute_subgraph_costs")
        edges_dict, prizes, costs, mapping = self.compute_subgraph_costs(
            edge_index, colls["nodes"].num_entities, prizes
        )

        # Retrieve the subgraph using the PCST algorithm
        logger.log(logging.INFO, "Running PCST algorithm")
        result_vertices, result_edges = pcst_fast.pcst_fast(
            edges_dict["edges"].tolist(),
            prizes.tolist(),
            costs.tolist(),
            self.root,
            self.num_clusters,
            self.pruning,
            self.verbosity_level,
        )

        # Get subgraph nodes and edges based on the result of the PCST algorithm
        logger.log(logging.INFO, "Getting subgraph nodes and edges")
        subgraph = self.get_subgraph_nodes_edges(
            colls["nodes"].num_entities,
            py.asarray(result_vertices),
            {
                "edges": py.asarray(result_edges),
                "num_prior_edges": edges_dict["num_prior_edges"],
                "edge_index": edge_index,
            },
            mapping,
        )
        print(subgraph)

        return subgraph
