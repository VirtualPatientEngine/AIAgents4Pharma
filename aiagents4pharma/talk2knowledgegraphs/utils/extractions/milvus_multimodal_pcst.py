"""
Exctraction of multimodal subgraph using Prize-Collecting Steiner Tree (PCST) algorithm.
"""

import asyncio
import logging
import platform
import subprocess
from typing import NamedTuple

import numpy as np
import pandas as pd
import pcst_fast
from pymilvus import Collection

try:
    import cudf  # type: ignore
    import cupy as cp  # type: ignore

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None
    cp = None

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemDetector:
    """Detect system capabilities and choose appropriate libraries."""

    def __init__(self):
        self.os_type = platform.system().lower()  # 'windows', 'linux', 'darwin'
        self.architecture = platform.machine().lower()  # 'x86_64', 'arm64', etc.
        self.has_nvidia_gpu = self._detect_nvidia_gpu()
        self.use_gpu = self.has_nvidia_gpu and self.os_type != "darwin"  # No CUDA on macOS

        logger.info("System Detection Results:")
        logger.info("  OS: %s", self.os_type)
        logger.info("  Architecture: %s", self.architecture)
        logger.info("  NVIDIA GPU detected: %s", self.has_nvidia_gpu)
        logger.info("  Will use GPU acceleration: %s", self.use_gpu)

    def _detect_nvidia_gpu(self) -> bool:
        """Detect if NVIDIA GPU is available."""
        try:
            # Try nvidia-smi command
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            return False

    def get_system_info(self) -> dict:
        """Get comprehensive system information."""
        return {
            "os_type": self.os_type,
            "architecture": self.architecture,
            "has_nvidia_gpu": self.has_nvidia_gpu,
            "use_gpu": self.use_gpu,
        }

    def is_gpu_compatible(self) -> bool:
        """Check if the system is compatible with GPU acceleration."""
        return self.has_nvidia_gpu and self.os_type != "darwin"


class DynamicLibraryLoader:
    """Dynamically load libraries based on system capabilities."""

    def __init__(self, detector: SystemDetector):
        self.detector = detector
        self.use_gpu = detector.use_gpu

        # Initialize attributes that will be set later
        self.py = None
        self.df = None
        self.pd = None
        self.np = None
        self.cudf = None
        self.cp = None

        # Import libraries based on system capabilities
        self._import_libraries()

        # Dynamic settings based on hardware
        self.normalize_vectors = self.use_gpu  # Only normalize for GPU
        self.metric_type = "IP" if self.use_gpu else "COSINE"

        logger.info("Library Configuration:")
        logger.info("  Using GPU acceleration: %s", self.use_gpu)
        logger.info("  Vector normalization: %s", self.normalize_vectors)
        logger.info("  Metric type: %s", self.metric_type)

    def _import_libraries(self):
        """Dynamically import libraries based on system capabilities."""
        # Set base libraries
        self.pd = pd
        self.np = np

        # Conditionally import GPU libraries
        if self.detector.use_gpu:
            if CUDF_AVAILABLE:
                self.cudf = cudf
                self.cp = cp
                self.py = cp  # Use cupy for array operations
                self.df = cudf  # Use cudf for dataframes
                logger.info("Successfully imported GPU libraries (cudf, cupy)")
            else:
                logger.error("cudf or cupy not found. Falling back to CPU mode.")
                self.detector.use_gpu = False
                self.use_gpu = False
                self._setup_cpu_mode()
        else:
            self._setup_cpu_mode()

    def _setup_cpu_mode(self):
        """Setup CPU mode with numpy and pandas."""
        self.py = self.np  # Use numpy for array operations
        self.df = self.pd  # Use pandas for dataframes
        self.normalize_vectors = False
        self.metric_type = "COSINE"
        logger.info("Using CPU mode with numpy and pandas")

    def normalize_matrix(self, matrix, axis: int = 1):
        """Normalize matrix using appropriate library."""
        if not self.normalize_vectors:
            return matrix

        if self.use_gpu:
            # Use cupy for GPU
            matrix_cp = self.cp.asarray(matrix).astype(self.cp.float32)
            norms = self.cp.linalg.norm(matrix_cp, axis=axis, keepdims=True)
            return matrix_cp / norms
        # CPU mode doesn't normalize for COSINE similarity
        return matrix

    def to_list(self, data):
        """Convert data to list format."""
        if hasattr(data, "tolist"):
            return data.tolist()
        if hasattr(data, "to_arrow"):
            return data.to_arrow().to_pylist()
        return list(data)


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
        use_description: Whether to use description embeddings.
        metric_type: The similarity metric type (dynamic based on hardware).
        loader: The dynamic library loader instance.
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
    metric_type: str = None  # Will be set dynamically
    loader: DynamicLibraryLoader = None

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

    async def load_edge_index_async(self, cfg: dict, _connection_manager=None) -> np.ndarray:
        """
        Load edge index using hybrid async/sync approach to avoid event loop issues.

        This method queries the edges collection to get head_index and tail_index,
        eliminating the need for pickle caching and reducing memory usage.

        Args:
            cfg: The configuration dictionary containing the Milvus setup.
            _connection_manager: Unused parameter for interface compatibility.

        Returns:
            numpy.ndarray: Edge index array with shape [2, num_edges]
        """
        logger.log(logging.INFO, "Loading edge index from Milvus collection (hybrid)")

        def load_edges_sync():
            """Load edges synchronously to avoid event loop issues."""

            collection_name = f"{cfg.milvus_db.database_name}_edges"
            edges_collection = Collection(name=collection_name)
            edges_collection.load()

            # Query all edges in batches
            batch_size = getattr(cfg.milvus_db, "query_batch_size", 10000)
            total_entities = edges_collection.num_entities
            logger.log(logging.INFO, "Total edges to process: %d", total_entities)

            head_list = []
            tail_list = []

            for start in range(0, total_entities, batch_size):
                end = min(start + batch_size, total_entities)
                logger.debug("Processing edge batch: %d to %d", start, end)

                batch = edges_collection.query(
                    expr=f"triplet_index >= {start} and triplet_index < {end}",
                    output_fields=["head_index", "tail_index"],
                )

                head_list.extend([r["head_index"] for r in batch])
                tail_list.extend([r["tail_index"] for r in batch])

            # Convert to numpy array format expected by PCST
            edge_index = self.loader.py.array([head_list, tail_list])
            logger.log(
                logging.INFO,
                "Edge index loaded (hybrid): shape %s",
                str(edge_index.shape),
            )

            return edge_index

        # Run in thread to avoid event loop conflicts
        return await asyncio.to_thread(load_edges_sync)

    def load_edge_index(self, cfg: dict) -> np.ndarray:
        """
        Load edge index synchronously from Milvus collection.

        This method queries the edges collection to get head_index and tail_index.

        Args:
            cfg: The configuration dictionary containing the Milvus setup.

        Returns:
            numpy.ndarray: Edge index array with shape [2, num_edges]
        """
        logger.log(logging.INFO, "Loading edge index from Milvus collection (sync)")

        collection_name = f"{cfg.milvus_db.database_name}_edges"
        edges_collection = Collection(name=collection_name)
        edges_collection.load()

        # Query all edges in batches
        batch_size = getattr(cfg.milvus_db, "query_batch_size", 10000)
        total_entities = edges_collection.num_entities
        logger.log(logging.INFO, "Total edges to process: %d", total_entities)

        head_list = []
        tail_list = []

        for start in range(0, total_entities, batch_size):
            end = min(start + batch_size, total_entities)
            logger.debug("Processing edge batch: %d to %d", start, end)

            batch = edges_collection.query(
                expr=f"triplet_index >= {start} and triplet_index < {end}",
                output_fields=["head_index", "tail_index"],
            )

            head_list.extend([r["head_index"] for r in batch])
            tail_list.extend([r["tail_index"] for r in batch])

        # Convert to numpy array format expected by PCST
        edge_index = self.loader.py.array([head_list, tail_list])
        logger.log(
            logging.INFO,
            "Edge index loaded (sync): shape %s",
            str(edge_index.shape),
        )

        return edge_index

    def _compute_node_prizes(self, query_emb: list, colls: dict) -> dict:
        """
        Compute the node prizes based on the similarity between the query and nodes.

        Args:
            query_emb: The query embedding. This can be an embedding of
                a prompt, sequence, or any other feature to be used for the subgraph extraction.
            colls: The collections of nodes, node-type specific nodes, and edges in Milvus.

        Returns:
            The prizes of the nodes.
        """
        # Initialize several variables
        topk = min(self.topk, colls["nodes"].num_entities)
        n_prizes = self.loader.py.zeros(colls["nodes"].num_entities, dtype=self.loader.py.float32)

        # Get the actual metric type to use
        actual_metric_type = self.metric_type or self.loader.metric_type

        # Calculate similarity for text features and update the score
        if self.use_description:
            # Search the collection with the text embedding
            res = colls["nodes"].search(
                data=[query_emb],
                anns_field="desc_emb",
                param={"metric_type": actual_metric_type},
                limit=topk,
                output_fields=["node_id"],
            )
        else:
            # Search the collection with the query embedding
            res = colls["nodes_type"].search(
                data=[query_emb],
                anns_field="feat_emb",
                param={"metric_type": actual_metric_type},
                limit=topk,
                output_fields=["node_id"],
            )

        # Update the prizes based on the search results
        n_prizes[[r.id for r in res[0]]] = self.loader.py.arange(topk, 0, -1).astype(
            self.loader.py.float32
        )

        return n_prizes

    async def _compute_node_prizes_async(
        self,
        query_emb: list,
        collection_name: str,
        connection_manager,
        use_description: bool = False,
    ) -> dict:
        """
        Compute the node prizes asynchronously using connection manager.

        Args:
            query_emb: The query embedding
            collection_name: Name of the collection to search
            connection_manager: The MilvusConnectionManager instance
            use_description: Whether to use description embeddings

        Returns:
            The prizes of the nodes
        """
        # Get collection stats for initialization
        stats = await connection_manager.async_get_collection_stats(collection_name)
        num_entities = stats["num_entities"]

        # Initialize prizes array
        topk = min(self.topk, num_entities)
        n_prizes = self.loader.py.zeros(num_entities, dtype=self.loader.py.float32)

        # Get the actual metric type to use
        actual_metric_type = self.metric_type or self.loader.metric_type

        # Determine search field based on use_description
        anns_field = "desc_emb" if use_description else "feat_emb"

        # Perform async search
        results = await connection_manager.async_search(
            collection_name=collection_name,
            data=[query_emb],
            anns_field=anns_field,
            param={"metric_type": actual_metric_type},
            limit=topk,
            output_fields=["node_id"],
        )

        # Update the prizes based on the search results
        if results and len(results) > 0:
            result_ids = [hit["id"] for hit in results[0]]
            n_prizes[result_ids] = self.loader.py.arange(topk, 0, -1).astype(self.loader.py.float32)

        return n_prizes

    def _compute_edge_prizes(self, text_emb: list, colls: dict):
        """
        Compute the edge prizes based on the similarity between the query and edges.

        Args:
            text_emb: The textual description embedding.
            colls: The collections of nodes, node-type specific nodes, and edges in Milvus.

        Returns:
            The prizes of the edges.
        """
        # Initialize several variables
        topk_e = min(self.topk_e, colls["edges"].num_entities)
        e_prizes = self.loader.py.zeros(colls["edges"].num_entities, dtype=self.loader.py.float32)

        # Get the actual metric type to use
        actual_metric_type = self.metric_type or self.loader.metric_type

        # Search the collection with the query embedding
        res = colls["edges"].search(
            data=[text_emb],
            anns_field="feat_emb",
            param={"metric_type": actual_metric_type},
            limit=topk_e,  # Only retrieve the top-k edges
            output_fields=["head_id", "tail_id"],
        )

        # Update the prizes based on the search results
        e_prizes[[r.id for r in res[0]]] = [r.score for r in res[0]]

        # Further process the edge_prizes
        unique_prizes, inverse_indices = self.loader.py.unique(e_prizes, return_inverse=True)
        topk_e_values = unique_prizes[self.loader.py.argsort(-unique_prizes)[:topk_e]]
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = inverse_indices == (unique_prizes == topk_e_values[k]).nonzero()[0]
            value = min((topk_e - k) / indices.sum().item(), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - self.c_const)

        return e_prizes

    async def _compute_edge_prizes_async(
        self, text_emb: list, collection_name: str, connection_manager
    ) -> dict:
        """
        Compute the edge prizes asynchronously using connection manager.

        Args:
            text_emb: The textual description embedding
            collection_name: Name of the edges collection
            connection_manager: The MilvusConnectionManager instance

        Returns:
            The prizes of the edges
        """
        # Get collection stats for initialization
        stats = await connection_manager.async_get_collection_stats(collection_name)
        num_entities = stats["num_entities"]

        # Initialize prizes array
        topk_e = min(self.topk_e, num_entities)
        e_prizes = self.loader.py.zeros(num_entities, dtype=self.loader.py.float32)

        # Get the actual metric type to use
        actual_metric_type = self.metric_type or self.loader.metric_type

        # Perform async search
        results = await connection_manager.async_search(
            collection_name=collection_name,
            data=[text_emb],
            anns_field="feat_emb",
            param={"metric_type": actual_metric_type},
            limit=topk_e,
            output_fields=["head_id", "tail_id"],
        )

        # Update the prizes based on the search results
        if results and len(results) > 0:
            result_ids = [hit["id"] for hit in results[0]]
            result_scores = [hit["distance"] for hit in results[0]]  # Use distance/score
            e_prizes[result_ids] = result_scores

        # Process edge prizes using helper method
        return self._process_edge_prizes(e_prizes, topk_e)

    def _process_edge_prizes(self, e_prizes, topk_e):
        """Helper method to process edge prizes and reduce complexity."""
        unique_prizes, inverse_indices = self.loader.py.unique(e_prizes, return_inverse=True)
        sorted_indices = self.loader.py.argsort(-unique_prizes)[:topk_e]
        topk_e_values = unique_prizes[sorted_indices]
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

    async def compute_prizes_async(
        self, text_emb: list, query_emb: list, cfg: dict, modality: str
    ) -> dict:
        """
        Compute node and edge prizes asynchronously in parallel using sync fallback.

        Args:
            text_emb: The textual description embedding
            query_emb: The query embedding
            cfg: The configuration dictionary containing the Milvus setup
            modality: The modality to use for the subgraph extraction

        Returns:
            The prizes of the nodes and edges
        """
        logger.log(logging.INFO, "Computing prizes in parallel (hybrid async/sync)")

        # Use existing sync method wrapped in asyncio.to_thread
        colls = self.prepare_collections(cfg, modality)
        return await asyncio.to_thread(self.compute_prizes, text_emb, query_emb, colls)

    def compute_subgraph_costs(self, edge_index, num_nodes: int, prizes: dict):
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
            self.loader.py.max(prizes["edges"]).item() * (1 - self.c_const / 2),
        )

        # Masks for real and virtual edges
        logger.log(logging.INFO, "Creating masks for real and virtual edges")
        real_["mask"] = prizes["edges"] <= updated_cost_e
        virt_["mask"] = ~real_["mask"]

        # Real edge indices
        logger.log(logging.INFO, "Computing real edges")
        real_["indices"] = self.loader.py.nonzero(real_["mask"])[0]
        real_["src"] = edge_index[0][real_["indices"]]
        real_["dst"] = edge_index[1][real_["indices"]]
        real_["edges"] = self.loader.py.stack([real_["src"], real_["dst"]], axis=1)
        real_["costs"] = updated_cost_e - prizes["edges"][real_["indices"]]

        # Edge index mapping: local real edge idx -> original global index
        logger.log(logging.INFO, "Creating mapping for real edges")
        mapping_edges = dict(
            zip(range(len(real_["indices"])), self.loader.to_list(real_["indices"]), strict=False)
        )

        # Virtual edge handling
        logger.log(logging.INFO, "Computing virtual edges")
        virt_["indices"] = self.loader.py.nonzero(virt_["mask"])[0]
        virt_["src"] = edge_index[0][virt_["indices"]]
        virt_["dst"] = edge_index[1][virt_["indices"]]
        virt_["prizes"] = prizes["edges"][virt_["indices"]] - updated_cost_e

        # Generate virtual node IDs
        logger.log(logging.INFO, "Generating virtual node IDs")
        virt_["num"] = virt_["indices"].shape[0]
        virt_["node_ids"] = self.loader.py.arange(num_nodes, num_nodes + virt_["num"])

        # Virtual edges: (src → virtual), (virtual → dst)
        logger.log(logging.INFO, "Creating virtual edges")
        virt_["edges_1"] = self.loader.py.stack([virt_["src"], virt_["node_ids"]], axis=1)
        virt_["edges_2"] = self.loader.py.stack([virt_["node_ids"], virt_["dst"]], axis=1)
        virt_["edges"] = self.loader.py.concatenate([virt_["edges_1"], virt_["edges_2"]], axis=0)
        virt_["costs"] = self.loader.py.zeros(
            (virt_["edges"].shape[0],), dtype=real_["costs"].dtype
        )

        # Combine real and virtual edges/costs
        logger.log(logging.INFO, "Combining real and virtual edges/costs")
        all_edges = self.loader.py.concatenate([real_["edges"], virt_["edges"]], axis=0)
        all_costs = self.loader.py.concatenate([real_["costs"], virt_["costs"]], axis=0)

        # Final prizes
        logger.log(logging.INFO, "Getting final prizes")
        final_prizes = self.loader.py.concatenate([prizes["nodes"], virt_["prizes"]], axis=0)

        # Mapping virtual node ID -> edge index in original graph
        logger.log(logging.INFO, "Creating mapping for virtual nodes")
        mapping_nodes = dict(
            zip(
                self.loader.to_list(virt_["node_ids"]),
                self.loader.to_list(virt_["indices"]),
                strict=False,
            )
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
        self, num_nodes: int, vertices, edges_dict: dict, mapping: dict
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

        # Retrieve the selected nodes and edges based on the given vertices and edges
        subgraph_nodes = vertices[vertices < num_nodes]
        subgraph_edges = [mapping["edges"][e.item()] for e in edges if e < num_prior_edges]
        virtual_vertices = vertices[vertices >= num_nodes]
        if len(virtual_vertices) > 0:
            virtual_edges = [mapping["nodes"][i.item()] for i in virtual_vertices]
            subgraph_edges = self.loader.py.array(subgraph_edges + virtual_edges)
        edge_index = edges_dict["edge_index"][:, subgraph_edges]
        subgraph_nodes = self.loader.py.unique(
            self.loader.py.concatenate([subgraph_nodes, edge_index[0], edge_index[1]])
        )

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

        # Load edge index directly from Milvus (replaces pickle cache)
        logger.log(logging.INFO, "Loading edge index from Milvus")
        edge_index = self.load_edge_index(cfg)

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
            self.loader.py.asarray(result_vertices),
            {
                "edges": self.loader.py.asarray(result_edges),
                "num_prior_edges": edges_dict["num_prior_edges"],
                "edge_index": edge_index,
            },
            mapping,
        )
        print(subgraph)

        return subgraph
