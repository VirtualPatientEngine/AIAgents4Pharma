#!/usr/bin/env python3
# pylint: skip-file
"""
Dynamic Cross-Platform PrimeKG Multimodal Data Loader for Milvus Database.
Automatically detects system capabilities and chooses appropriate libraries and configurations.

Supports:
- Windows, Linux, macOS
- CPU-only mode (pandas/numpy)
- NVIDIA GPU mode (cudf/cupy)
- Dynamic index selection based on hardware
- Automatic dependency installation
"""

import glob
import logging
import os
import platform
import subprocess
import sys
from typing import Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="[DATA LOADER] %(message)s")
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
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            return False

    def get_required_packages(self) -> list[str]:
        """Get list of packages to install based on system capabilities - matches original logic."""
        if self.use_gpu and self.os_type == "linux":
            # Exact package list from original script for GPU mode
            packages = [
                "pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12",
                "pip install --extra-index-url=https://pypi.nvidia.com dask-cudf-cu12",
                "pip install pymilvus",
                "pip install numpy",
                "pip install pandas",
                "pip install tqdm",
            ]
            return packages
        else:
            # CPU-only packages
            packages = [
                "pip install pymilvus",
                "pip install numpy",
                "pip install pandas",
                "pip install tqdm",
                "pip install pyarrow",
            ]
            return packages

    def install_packages(self):
        """Install required packages using original script's exact logic."""
        packages = self.get_required_packages()

        logger.info(
            "Installing packages for %s system%s",
            self.os_type,
            " with GPU support" if self.use_gpu else "",
        )

        for package_cmd in packages:
            logger.info("Running: %s", package_cmd)
            try:
                result = subprocess.run(
                    package_cmd.split(),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    logger.error("Error installing package: %s", result.stderr)
                    if "cudf" in package_cmd or "dask-cudf" in package_cmd:
                        logger.warning("GPU package installation failed, falling back to CPU mode")
                        self.use_gpu = False
                        return self.install_packages()  # Retry with CPU packages
                    else:
                        sys.exit(1)
                else:
                    logger.info("Successfully installed: %s", package_cmd.split()[-1])
            except subprocess.CalledProcessError:
                logger.error("Failed to install package: %s", package_cmd.split()[-1])
                if "cudf" in package_cmd:
                    logger.warning("GPU package installation failed, falling back to CPU mode")
                    self.use_gpu = False
                    return self.install_packages()  # Retry with CPU packages
                else:
                    raise
            except subprocess.TimeoutExpired:
                logger.error("Installation timeout for package: %s", package_cmd)
                raise


class DynamicDataLoader:
    """Dynamic data loader that adapts to system capabilities."""

    def __init__(self, config: dict[str, Any]):
        """Initialize with system detection and dynamic library loading."""
        self.config = config
        self.detector = SystemDetector()

        # Install packages if needed
        if config.get("auto_install_packages", True):
            self.detector.install_packages()

        # Import libraries based on system capabilities
        self._import_libraries()

        # Configuration - exact original parameters
        self.milvus_host = config.get("milvus_host", "localhost")
        self.milvus_port = config.get("milvus_port", "19530")
        self.milvus_user = config.get("milvus_user", "root")
        self.milvus_password = config.get("milvus_password", "Milvus")
        self.milvus_database = config.get("milvus_database", "t2kg_primekg")
        self.data_dir = config.get("data_dir", "./data")
        self.batch_size = config.get("batch_size", 500)
        self.chunk_size = config.get("chunk_size", 5)  # Original chunk_size parameter

        # Dynamic settings based on hardware
        self.use_gpu = self.detector.use_gpu
        self.normalize_vectors = self.use_gpu  # Only normalize for GPU (original logic)
        self.vector_index_type = "GPU_CAGRA" if self.use_gpu else "HNSW"
        self.metric_type = "IP" if self.use_gpu else "COSINE"

        logger.info("Loader Configuration:")
        logger.info("  Using GPU acceleration: %s", self.use_gpu)
        logger.info("  Vector normalization: %s", self.normalize_vectors)
        logger.info("  Vector index type: %s", self.vector_index_type)
        logger.info("  Metric type: %s", self.metric_type)
        logger.info("  Data directory: %s", self.data_dir)
        logger.info("  Batch size: %s", self.batch_size)
        logger.info("  Chunk size: %s", self.chunk_size)

    def _import_libraries(self):
        """Dynamically import libraries - matches original script's import logic."""
        # Always import base libraries
        import numpy as np
        import pandas as pd
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            db,
            utility,
        )
        from tqdm import tqdm

        self.pd = pd
        self.np = np
        self.tqdm = tqdm
        self.pymilvus_modules = {
            "db": db,
            "connections": connections,
            "FieldSchema": FieldSchema,
            "CollectionSchema": CollectionSchema,
            "DataType": DataType,
            "Collection": Collection,
            "utility": utility,
        }

        # Conditionally import GPU libraries - matches original error handling
        if self.detector.use_gpu:
            try:
                import cudf  # pyright: ignore
                import cupy as cp  # pyright: ignore

                self.cudf = cudf
                self.cp = cp
                logger.info("Successfully imported GPU libraries (cudf, cupy)")
            except ImportError:
                logger.error(
                    "[DATA LOADER] cudf or cupy not found. "
                    "Please ensure they are installed correctly."
                )
                logger.error("Import error occurred - GPU libraries not available")
                # Match original script's exit behavior for critical GPU import failure
                if not os.getenv("FORCE_CPU", "false").lower() == "true":
                    logger.error(
                        "GPU libraries required but not available. "
                        "Set FORCE_CPU=true to use CPU mode."
                    )
                    sys.exit(1)
                else:
                    logger.warning("Falling back to CPU mode due to FORCE_CPU=true")
                    self.detector.use_gpu = False
                    self.use_gpu = False

    def _read_dataframe(
        self, file_path: str, columns: list[str] | None = None
    ) -> Union["pd.DataFrame", "cudf.DataFrame"]:  # type: ignore[reportUndefinedVariable]  # noqa: F821
        """Read dataframe using appropriate library."""
        if self.use_gpu:
            return self.cudf.read_parquet(file_path, columns=columns)
        else:
            return self.pd.read_parquet(file_path, columns=columns)

    def _concat_dataframes(
        self, df_list: list, ignore_index: bool = True
    ) -> Union["pd.DataFrame", "cudf.DataFrame"]:  # type: ignore[reportUndefinedVariable]  # noqa: F821
        """Concatenate dataframes using appropriate library."""
        if self.use_gpu:
            return self.cudf.concat(df_list, ignore_index=ignore_index)
        else:
            return self.pd.concat(df_list, ignore_index=ignore_index)

    def _normalize_matrix(self, matrix, axis: int = 1):
        """Normalize matrix using appropriate library."""
        if not self.normalize_vectors:
            return matrix

        if self.use_gpu:
            # Use cupy for GPU
            matrix_cp = self.cp.asarray(matrix).astype(self.cp.float32)
            norms = self.cp.linalg.norm(matrix_cp, axis=axis, keepdims=True)
            return matrix_cp / norms
        else:
            # Use numpy for CPU (but we don't normalize for CPU/COSINE)
            return matrix

    def _extract_embeddings(self, df, column_name: str):
        """Extract embeddings and convert to appropriate format."""
        if self.use_gpu:
            # cuDF list extraction
            emb_data = self.cp.asarray(df[column_name].list.leaves).astype(self.cp.float32)
            return emb_data.reshape(df.shape[0], -1)
        else:
            # pandas extraction
            emb_list = []
            for emb in df[column_name]:
                if isinstance(emb, list):
                    emb_list.append(emb)
                else:
                    emb_list.append(emb.tolist() if hasattr(emb, "tolist") else emb)
            return self.np.array(emb_list, dtype=self.np.float32)

    def _to_list(self, data):
        """Convert data to list format for Milvus insertion."""
        if self.use_gpu:
            # For cuDF data, use to_arrow().to_pylist()
            if hasattr(data, "to_arrow"):
                return data.to_arrow().to_pylist()
            elif hasattr(data, "tolist"):
                # Fallback for cupy arrays
                return data.tolist()
            else:
                return list(data)
        else:
            # For pandas/numpy data
            if hasattr(data, "tolist"):
                return data.tolist()
            elif hasattr(data, "to_arrow"):
                return data.to_arrow().to_pylist()
            else:
                return list(data)

    def connect_to_milvus(self):
        """Connect to Milvus and setup database."""
        logger.info("Connecting to Milvus at %s:%s", self.milvus_host, self.milvus_port)

        self.pymilvus_modules["connections"].connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
            user=self.milvus_user,
            password=self.milvus_password,
        )

        # Check if database exists, create if it doesn't
        if self.milvus_database not in self.pymilvus_modules["db"].list_database():
            logger.info("Creating database: %s", self.milvus_database)
            self.pymilvus_modules["db"].create_database(self.milvus_database)

        # Switch to the desired database
        self.pymilvus_modules["db"].using_database(self.milvus_database)
        logger.info("Using database: %s", self.milvus_database)

    def load_graph_data(self):
        """Load the parquet files containing graph data."""
        logger.info("Loading graph data from: %s", self.data_dir)

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        graph = {}
        for element in ["nodes", "edges"]:
            graph[element] = {}
            for stage in ["enrichment", "embedding"]:
                logger.info("Processing %s %s", element, stage)

                file_list = glob.glob(os.path.join(self.data_dir, element, stage, "*.parquet.gzip"))
                logger.info("Found %d files for %s %s", len(file_list), element, stage)

                if not file_list:
                    logger.warning("No files found for %s %s", element, stage)
                    continue

                # For edges embedding, process in chunks due to size
                if element == "edges" and stage == "embedding":
                    chunk_size = self.chunk_size
                    graph[element][stage] = []
                    for i in range(0, len(file_list), chunk_size):
                        chunk_files = file_list[i : i + chunk_size]
                        chunk_df_list = []
                        for f in chunk_files:
                            df = self._read_dataframe(f, columns=["triplet_index", "edge_emb"])
                            chunk_df_list.append(df)
                        chunk_df = self._concat_dataframes(chunk_df_list, ignore_index=True)
                        graph[element][stage].append(chunk_df)
                else:
                    # For other combinations, read all files
                    df_list = []
                    for f in file_list:
                        df = self._read_dataframe(f)
                        df_list.append(df)
                    graph[element][stage] = self._concat_dataframes(df_list, ignore_index=True)

        logger.info("Graph data loaded successfully")
        return graph

    def _get_embedding_dimension(self, df, column_name: str) -> int:
        """Get embedding dimension using original script's exact logic."""
        first_emb = df.iloc[0][column_name]
        if self.use_gpu:
            # cuDF format - matches original:
            # len(nodes_df.iloc[0]['desc_emb'].to_arrow().to_pylist()[0])
            return len(first_emb.to_arrow().to_pylist()[0])
        else:
            # pandas format
            if isinstance(first_emb, list):
                return len(first_emb)
            else:
                return len(first_emb.tolist() if hasattr(first_emb, "tolist") else first_emb)

    def create_nodes_collection(self, nodes_df):
        """Create and populate the main nodes collection."""
        logger.info("Creating main nodes collection...")
        node_coll_name = f"{self.milvus_database}_nodes"

        # Get embedding dimension
        emb_dim = self._get_embedding_dimension(nodes_df, "desc_emb")

        node_fields = [
            self.pymilvus_modules["FieldSchema"](
                name="node_index",
                dtype=self.pymilvus_modules["DataType"].INT64,
                is_primary=True,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="node_id",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="node_name",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
                enable_analyzer=True,
                enable_match=True,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="node_type",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
                enable_analyzer=True,
                enable_match=True,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="desc",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=40960,
                enable_analyzer=True,
                enable_match=True,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="desc_emb",
                dtype=self.pymilvus_modules["DataType"].FLOAT_VECTOR,
                dim=emb_dim,
            ),
        ]

        schema = self.pymilvus_modules["CollectionSchema"](
            fields=node_fields, description=f"Schema for collection {node_coll_name}"
        )

        # Create collection if it doesn't exist
        if not self.pymilvus_modules["utility"].has_collection(node_coll_name):
            collection = self.pymilvus_modules["Collection"](name=node_coll_name, schema=schema)
        else:
            collection = self.pymilvus_modules["Collection"](name=node_coll_name)

        # Create indexes with dynamic parameters
        collection.create_index(
            field_name="node_index",
            index_params={"index_type": "STL_SORT"},
            index_name="node_index_index",
        )
        collection.create_index(
            field_name="node_name",
            index_params={"index_type": "INVERTED"},
            index_name="node_name_index",
        )
        collection.create_index(
            field_name="node_type",
            index_params={"index_type": "INVERTED"},
            index_name="node_type_index",
        )
        collection.create_index(
            field_name="desc",
            index_params={"index_type": "INVERTED"},
            index_name="desc_index",
        )
        collection.create_index(
            field_name="desc_emb",
            index_params={
                "index_type": self.vector_index_type,
                "metric_type": self.metric_type,
            },
            index_name="desc_emb_index",
        )

        # Prepare and insert data
        desc_emb_data = self._extract_embeddings(nodes_df, "desc_emb")
        desc_emb_normalized = self._normalize_matrix(desc_emb_data, axis=1)

        data = [
            self._to_list(nodes_df["node_index"]),
            self._to_list(nodes_df["node_id"]),
            self._to_list(nodes_df["node_name"]),
            self._to_list(nodes_df["node_type"]),
            self._to_list(nodes_df["desc"]),
            self._to_list(desc_emb_normalized),
        ]

        # Insert data in batches
        total = len(data[0])
        for i in self.tqdm(range(0, total, self.batch_size), desc="Inserting nodes"):
            batch = [col[i : i + self.batch_size] for col in data]
            collection.insert(batch)

        collection.flush()
        logger.info("Nodes collection created with %d entities", collection.num_entities)

    def create_node_type_collections(self, nodes_df):
        """Create separate collections for each node type."""
        logger.info("Creating node type-specific collections...")

        for node_type, nodes_df_ in self.tqdm(
            nodes_df.groupby("node_type"), desc="Processing node types"
        ):
            node_coll_name = f"{self.milvus_database}_nodes_{node_type.replace('/', '_')}"

            # Get embedding dimensions
            desc_dim = self._get_embedding_dimension(nodes_df_, "desc_emb")
            feat_dim = self._get_embedding_dimension(nodes_df_, "feat_emb")

            node_fields = [
                self.pymilvus_modules["FieldSchema"](
                    name="node_index",
                    dtype=self.pymilvus_modules["DataType"].INT64,
                    is_primary=True,
                    auto_id=False,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="node_id",
                    dtype=self.pymilvus_modules["DataType"].VARCHAR,
                    max_length=1024,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="node_name",
                    dtype=self.pymilvus_modules["DataType"].VARCHAR,
                    max_length=1024,
                    enable_analyzer=True,
                    enable_match=True,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="node_type",
                    dtype=self.pymilvus_modules["DataType"].VARCHAR,
                    max_length=1024,
                    enable_analyzer=True,
                    enable_match=True,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="desc",
                    dtype=self.pymilvus_modules["DataType"].VARCHAR,
                    max_length=40960,
                    enable_analyzer=True,
                    enable_match=True,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="desc_emb",
                    dtype=self.pymilvus_modules["DataType"].FLOAT_VECTOR,
                    dim=desc_dim,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="feat",
                    dtype=self.pymilvus_modules["DataType"].VARCHAR,
                    max_length=40960,
                    enable_analyzer=True,
                    enable_match=True,
                ),
                self.pymilvus_modules["FieldSchema"](
                    name="feat_emb",
                    dtype=self.pymilvus_modules["DataType"].FLOAT_VECTOR,
                    dim=feat_dim,
                ),
            ]

            schema = self.pymilvus_modules["CollectionSchema"](
                fields=node_fields,
                description=f"schema for collection {node_coll_name}",
            )

            if not self.pymilvus_modules["utility"].has_collection(node_coll_name):
                collection = self.pymilvus_modules["Collection"](name=node_coll_name, schema=schema)
            else:
                collection = self.pymilvus_modules["Collection"](name=node_coll_name)

            # Create indexes with dynamic parameters
            collection.create_index(
                field_name="node_index",
                index_params={"index_type": "STL_SORT"},
                index_name="node_index_index",
            )
            collection.create_index(
                field_name="node_name",
                index_params={"index_type": "INVERTED"},
                index_name="node_name_index",
            )
            collection.create_index(
                field_name="node_type",
                index_params={"index_type": "INVERTED"},
                index_name="node_type_index",
            )
            collection.create_index(
                field_name="desc",
                index_params={"index_type": "INVERTED"},
                index_name="desc_index",
            )
            collection.create_index(
                field_name="desc_emb",
                index_params={
                    "index_type": self.vector_index_type,
                    "metric_type": self.metric_type,
                },
                index_name="desc_emb_index",
            )
            collection.create_index(
                field_name="feat_emb",
                index_params={
                    "index_type": self.vector_index_type,
                    "metric_type": self.metric_type,
                },
                index_name="feat_emb_index",
            )

            # Prepare data
            desc_emb_data = self._extract_embeddings(nodes_df_, "desc_emb")
            feat_emb_data = self._extract_embeddings(nodes_df_, "feat_emb")

            desc_emb_normalized = self._normalize_matrix(desc_emb_data, axis=1)
            feat_emb_normalized = self._normalize_matrix(feat_emb_data, axis=1)

            data = [
                self._to_list(nodes_df_["node_index"]),
                self._to_list(nodes_df_["node_id"]),
                self._to_list(nodes_df_["node_name"]),
                self._to_list(nodes_df_["node_type"]),
                self._to_list(nodes_df_["desc"]),
                self._to_list(desc_emb_normalized),
                self._to_list(nodes_df_["feat"]),
                self._to_list(feat_emb_normalized),
            ]

            # Insert data in batches
            total_rows = len(data[0])
            for i in range(0, total_rows, self.batch_size):
                batch = [col[i : i + self.batch_size] for col in data]
                collection.insert(batch)

            collection.flush()
            logger.info(
                "Collection %s created with %d entities",
                node_coll_name,
                collection.num_entities,
            )

    def create_edges_collection(self, edges_enrichment_df, edges_embedding_df: list):
        """Create and populate the edges collection - exact original logic."""
        logger.info("Creating edges collection...")

        edge_coll_name = f"{self.milvus_database}_edges"

        # Get embedding dimension from first chunk - exact original logic
        if self.use_gpu:
            emb_dim = len(edges_embedding_df[0].loc[0, "edge_emb"])  # Original cudf access
        else:
            first_edge_emb = edges_embedding_df[0].iloc[0]["edge_emb"]
            emb_dim = (
                len(first_edge_emb)
                if isinstance(first_edge_emb, list)
                else len(first_edge_emb.tolist())
            )

        edge_fields = [
            self.pymilvus_modules["FieldSchema"](
                name="triplet_index",
                dtype=self.pymilvus_modules["DataType"].INT64,
                is_primary=True,
                auto_id=False,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="head_id",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="head_index", dtype=self.pymilvus_modules["DataType"].INT64
            ),
            self.pymilvus_modules["FieldSchema"](
                name="tail_id",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="tail_index", dtype=self.pymilvus_modules["DataType"].INT64
            ),
            self.pymilvus_modules["FieldSchema"](
                name="edge_type",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="display_relation",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=1024,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="feat",
                dtype=self.pymilvus_modules["DataType"].VARCHAR,
                max_length=40960,
            ),
            self.pymilvus_modules["FieldSchema"](
                name="feat_emb",
                dtype=self.pymilvus_modules["DataType"].FLOAT_VECTOR,
                dim=emb_dim,
            ),
        ]

        edge_schema = self.pymilvus_modules["CollectionSchema"](
            fields=edge_fields, description="Schema for edges collection"
        )

        if not self.pymilvus_modules["utility"].has_collection(edge_coll_name):
            collection = self.pymilvus_modules["Collection"](
                name=edge_coll_name, schema=edge_schema
            )
        else:
            collection = self.pymilvus_modules["Collection"](name=edge_coll_name)

        # Create indexes with dynamic parameters
        collection.create_index(
            field_name="triplet_index",
            index_params={"index_type": "STL_SORT"},
            index_name="triplet_index_index",
        )
        collection.create_index(
            field_name="head_index",
            index_params={"index_type": "STL_SORT"},
            index_name="head_index_index",
        )
        collection.create_index(
            field_name="tail_index",
            index_params={"index_type": "STL_SORT"},
            index_name="tail_index_index",
        )
        collection.create_index(
            field_name="feat_emb",
            index_params={
                "index_type": self.vector_index_type,
                "metric_type": self.metric_type,
            },
            index_name="feat_emb_index",
        )

        # Iterate over chunked edges embedding df - exact original logic
        for edges_df in self.tqdm(edges_embedding_df, desc="Processing edge chunks"):
            # Merge enrichment with embedding
            merged_edges_df = edges_enrichment_df.merge(
                edges_df[["triplet_index", "edge_emb"]], on="triplet_index", how="inner"
            )

            # Prepare embeddings - exact original logic for GPU
            if self.use_gpu:
                edge_emb_cp = (
                    self.cp.asarray(merged_edges_df["edge_emb"].list.leaves)
                    .astype(self.cp.float32)
                    .reshape(merged_edges_df.shape[0], -1)
                )
                edge_emb_norm = self._normalize_matrix(edge_emb_cp, axis=1)
            else:
                edge_emb_data = self._extract_embeddings(merged_edges_df, "edge_emb")
                edge_emb_norm = self._normalize_matrix(edge_emb_data, axis=1)

            data = [
                self._to_list(merged_edges_df["triplet_index"]),
                self._to_list(merged_edges_df["head_id"]),
                self._to_list(merged_edges_df["head_index"]),
                self._to_list(merged_edges_df["tail_id"]),
                self._to_list(merged_edges_df["tail_index"]),
                self._to_list(merged_edges_df["edge_type_str"]),  # Original field name
                self._to_list(merged_edges_df["display_relation"]),
                self._to_list(merged_edges_df["feat"]),
                self._to_list(edge_emb_norm),
            ]

            # Insert data in batches
            total = len(data[0])
            for i in self.tqdm(range(0, total, self.batch_size), desc="Inserting edges"):
                batch_data = [d[i : i + self.batch_size] for d in data]
                collection.insert(batch_data)

        collection.flush()
        logger.info("Edges collection created with %d entities", collection.num_entities)

    def run(self):
        """Main execution method."""
        try:
            logger.info("Starting Dynamic Milvus data loading process...")
            logger.info("System: %s %s", self.detector.os_type, self.detector.architecture)
            logger.info("GPU acceleration: %s", self.use_gpu)

            # Connect to Milvus
            self.connect_to_milvus()

            # Load graph data
            graph = self.load_graph_data()

            # Prepare data
            logger.info("Data Preparation started...")
            # Get nodes enrichment and embedding dataframes
            nodes_enrichment_df = graph["nodes"]["enrichment"]
            nodes_embedding_df = graph["nodes"]["embedding"]

            # Get edges enrichment and embedding dataframes
            edges_enrichment_df = graph["edges"]["enrichment"]
            edges_embedding_df = graph["edges"]["embedding"]  # List of dataframes

            # Merge nodes enrichment and embedding dataframes
            merged_nodes_df = nodes_enrichment_df.merge(
                nodes_embedding_df[["node_id", "desc_emb", "feat_emb"]],
                on="node_id",
                how="left",
            )

            # Create collections and load data
            self.create_nodes_collection(merged_nodes_df)
            self.create_node_type_collections(merged_nodes_df)
            self.create_edges_collection(edges_enrichment_df, edges_embedding_df)

            # List all collections for verification
            logger.info("Data loading completed successfully!")
            logger.info("Created collections:")
            for coll in self.pymilvus_modules["utility"].list_collections():
                collection = self.pymilvus_modules["Collection"](name=coll)
                logger.info("  %s: %d entities", coll, collection.num_entities)

        except Exception:
            logger.error("Error occurred during data loading")
            logger.debug("Detailed error information available in debug mode")
            raise


def main():
    """Main function to run the dynamic data loader."""
    # Resolve the fallback data path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "tests/files/biobridge_multimodal/")

    # Configuration with environment variable fallbacks - matches original exactly
    config = {
        "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
        "milvus_port": os.getenv("MILVUS_PORT", "19530"),
        "milvus_user": os.getenv("MILVUS_USER", "root"),
        "milvus_password": os.getenv("MILVUS_PASSWORD", "Milvus"),
        "milvus_database": os.getenv("MILVUS_DATABASE", "t2kg_primekg"),
        "data_dir": os.getenv("DATA_DIR", default_data_dir),
        "batch_size": int(os.getenv("BATCH_SIZE", "500")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "5")),
        "auto_install_packages": os.getenv("AUTO_INSTALL_PACKAGES", "true").lower() == "true",
    }

    # Override detection for testing/forcing specific modes
    force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
    if force_cpu:
        logger.info("FORCE_CPU environment variable set - forcing CPU mode")

    # Print configuration for debugging - matches original format
    logger.info("=== Dynamic Milvus Data Loader ===")
    logger.info("Configuration:")
    for key, value in config.items():
        # Don't log sensitive information
        if any(
            sensitive in key.lower() for sensitive in ["password", "user", "token", "key", "secret"]
        ):
            logger.info("  %s: %s", key, "*" * min(8, len(str(value))))
        else:
            logger.info("  %s: %s", key, value)

    # Additional environment info
    logger.info("Environment:")
    logger.info("  Python version: %s", sys.version)
    logger.info("  Platform: %s", platform.platform())
    logger.info("  Force CPU mode: %s", force_cpu)
    logger.info("  Script directory: %s", script_dir)
    logger.info("  Default data directory: %s", default_data_dir)

    try:
        # Create and run dynamic data loader
        loader = DynamicDataLoader(config)

        # Override GPU detection if forced
        if force_cpu:
            loader.detector.use_gpu = False
            loader.use_gpu = False
            loader.normalize_vectors = False
            loader.vector_index_type = "HNSW"
            loader.metric_type = "COSINE"
            logger.info("Forced CPU mode - updated loader settings")

        # Run the data loading process
        loader.run()

        logger.info("=== Data Loading Completed Successfully ===")

    except KeyboardInterrupt:
        logger.info("Data loading interrupted by user")
        sys.exit(1)
    except Exception:
        logger.error("Fatal error occurred during data loading")
        logger.debug("Detailed error information available in debug mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
