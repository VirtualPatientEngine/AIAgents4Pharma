# pylint: disable=wrong-import-position
#!/usr/bin/env python3
"""
Script to load PrimeKG multimodal data into Milvus database.
This script runs after Milvus container is ready and loads the .pkl file data.
"""

import os
import sys
import subprocess
import glob
import logging
from typing import Dict, Any, List

def install_packages():
    """Install required packages."""
    packages = [
        "pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12",
        "pip install --extra-index-url=https://pypi.nvidia.com dask-cudf-cu12",
        "pip install pymilvus==2.5.11",
        "pip install numpy==1.26.4",
        "pip install pandas==2.1.3",
        "pip install tqdm==4.67.1",
    ]

    print("[DATA LOADER] Installing required packages...")
    for package_cmd in packages:
        print(f"[DATA LOADER] Running: {package_cmd}")
        result = subprocess.run(package_cmd.split(), capture_output=True, text=True, check=True)
        if result.returncode != 0:
            print(f"[DATA LOADER] Error installing package: {result.stderr}")
            sys.exit(1)
    print("[DATA LOADER] All packages installed successfully!")

# Install packages first
install_packages()

try:
    import cudf
    import cupy as cp
except ImportError as e:
    print("[DATA LOADER] cudf or cupy not found. Please ensure they are installed correctly.")
    sys.exit(1)

from pymilvus import (
    db,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='[DATA LOADER] %(message)s')
logger = logging.getLogger(__name__)

class MilvusDataLoader:
    """
    Class to handle loading of BioBridge-PrimeKG multimodal data into Milvus.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MilvusDataLoader with configuration parameters."""
        self.config = config
        self.milvus_host = config.get('milvus_host', 'localhost')
        self.milvus_port = config.get('milvus_port', '19530')
        self.milvus_user = config.get('milvus_user', 'root')
        self.milvus_password = config.get('milvus_password', 'Milvus')
        self.milvus_database = config.get('milvus_database', 't2kg_primekg')
        self.data_dir = config.get('data_dir',
                                   'tests/files/biobridge_multimodal/')
        self.batch_size = config.get('batch_size', 500)
        self.chunk_size = config.get('chunk_size', 5)

    def normalize_matrix(self, m, axis=1):
        """Normalize each row of a 2D matrix using CuPy."""
        norms = cp.linalg.norm(m, axis=axis, keepdims=True)
        return m / norms

    def normalize_vector(self, v):
        """Normalize a vector using CuPy."""
        v = cp.asarray(v)
        norm = cp.linalg.norm(v)
        return v / norm

    def connect_to_milvus(self):
        """Connect to Milvus and setup database."""
        logger.info("Connecting to Milvus at %s:%s", self.milvus_host, self.milvus_port)

        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
            user=self.milvus_user,
            password=self.milvus_password
        )

        # Check if database exists, create if it doesn't
        if self.milvus_database not in db.list_database():
            logger.info("Creating database: %s", self.milvus_database)
            db.create_database(self.milvus_database)

        # Switch to the desired database
        db.using_database(self.milvus_database)
        logger.info("Using database: %s", self.milvus_database)

    def load_graph_data(self):
        """Load the pickle file containing graph data."""
        logger.info("Loading graph data from: %s", self.data_dir)

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load dataframes containing nodes and edges
        # Loop over nodes and edges
        graph = {}
        for element in ["nodes", "edges"]:
            # Make an empty dictionary for each folder
            graph[element] = {}
            for stage in ["enrichment", "embedding"]:
                print(element, stage)
                # Create the file pattern for the current subfolder
                file_list = glob.glob(os.path.join(self.data_dir,
                                                   element,
                                                   stage,
                                                   '*.parquet.gzip'))
                print(file_list)
                # Read and concatenate all dataframes in the folder
                # Except the edges embedding, which is too large to read in one go
                # We are using a chunk size to read the edges embedding in smaller parts instead
                if element == "edges" and stage == "embedding":
                    # For edges embedding, only read two columns:
                    # triplet_index and edge_emb
                    # Loop by chunks
                    chunk_size = self.chunk_size
                    graph[element][stage] = []
                    for i in range(0, len(file_list), chunk_size):
                        chunk_files = file_list[i:i+chunk_size]
                        chunk_df = cudf.concat([
                            cudf.read_parquet(f, columns=["triplet_index", "edge_emb"])
                            for f in chunk_files
                        ], ignore_index=True)
                        graph[element][stage].append(chunk_df)
                else:
                    # For nodes and edges enrichment,
                    # read and concatenate all dataframes in the folder
                    # This includes the nodes embedding,
                    # which is small enough to read in one go
                    graph[element][stage] = cudf.concat([
                        cudf.read_parquet(f) for f in file_list
                    ], ignore_index=True)

        logger.info("Graph data loaded successfully")
        return graph

    def create_nodes_collection(self, nodes_df: cudf.DataFrame):
        """Create and populate the main nodes collection."""
        logger.info("Creating main nodes collection...")
        node_coll_name = f"{self.milvus_database}_nodes"

        node_fields = [
            FieldSchema(name="node_index",
                        dtype=DataType.INT64,
                        is_primary=True),
            FieldSchema(name="node_id",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="node_name",
                        dtype=DataType.VARCHAR,
                        max_length=1024,
                        enable_analyzer=True,
                        enable_match=True),
            FieldSchema(name="node_type",
                        dtype=DataType.VARCHAR,
                        max_length=1024,
                        enable_analyzer=True,
                        enable_match=True),
            FieldSchema(name="desc",
                        dtype=DataType.VARCHAR,
                        max_length=40960,
                        enable_analyzer=True,
                        enable_match=True),
            FieldSchema(name="desc_emb",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=len(nodes_df.iloc[0]['desc_emb'].to_arrow().to_pylist()[0])),
        ]
        schema = CollectionSchema(fields=node_fields,
                                  description=f"Schema for collection {node_coll_name}")

        # Create collection if it doesn't exist
        if not utility.has_collection(node_coll_name):
            collection = Collection(name=node_coll_name, schema=schema)
        else:
            collection = Collection(name=node_coll_name)

        # Create indexes
        collection.create_index(field_name="node_index",
                                index_params={"index_type": "STL_SORT"},
                                index_name="node_index_index")
        collection.create_index(field_name="node_name",
                                index_params={"index_type": "INVERTED"},
                                index_name="node_name_index")
        collection.create_index(field_name="node_type",
                                index_params={"index_type": "INVERTED"},
                                index_name="node_type_index")
        collection.create_index(field_name="desc",
                                index_params={"index_type": "INVERTED"},
                                index_name="desc_index")
        collection.create_index(field_name="desc_emb",
                                index_params={"index_type": "GPU_CAGRA",
                                              "metric_type": "IP"},
                                index_name="desc_emb_index")

        # Prepare and insert data
        desc_emb_norm = cp.asarray(nodes_df["desc_emb"].list.leaves).astype(cp.float32).\
            reshape(nodes_df.shape[0], -1)
        desc_emb_norm = self.normalize_matrix(desc_emb_norm, axis=1)
        data = [
            nodes_df["node_index"].to_arrow().to_pylist(),
            nodes_df["node_id"].to_arrow().to_pylist(),
            nodes_df["node_name"].to_arrow().to_pylist(),
            nodes_df["node_type"].to_arrow().to_pylist(),
            nodes_df["desc"].to_arrow().to_pylist(),
            desc_emb_norm.tolist(), # Use normalized embeddings
        ]

        # Insert data in batches
        total = len(data[0])
        for i in tqdm(range(0, total, self.batch_size), desc="Inserting nodes"):
            batch = [col[i:i+self.batch_size] for col in data]
            collection.insert(batch)

        collection.flush()
        logger.info("Nodes collection created with %d entities", collection.num_entities)

    def create_node_type_collections(self, nodes_df: cudf.DataFrame):
        """Create separate collections for each node type."""
        logger.info("Creating node type-specific collections...")

        for node_type, nodes_df_ in tqdm(nodes_df.groupby('node_type'),
                                         desc="Processing node types"):
            node_coll_name = f"{self.milvus_database}_nodes_{node_type.replace('/', '_')}"

            node_fields = [
                FieldSchema(name="node_index",
                            dtype=DataType.INT64,
                            is_primary=True,
                            auto_id=False),
                FieldSchema(name="node_id",
                            dtype=DataType.VARCHAR,
                            max_length=1024),
                FieldSchema(name="node_name",
                            dtype=DataType.VARCHAR,
                            max_length=1024,
                            enable_analyzer=True,
                            enable_match=True),
                FieldSchema(name="node_type",
                            dtype=DataType.VARCHAR,
                            max_length=1024,
                            enable_analyzer=True,
                            enable_match=True),
                FieldSchema(name="desc",
                            dtype=DataType.VARCHAR,
                            max_length=40960,
                            enable_analyzer=True,
                            enable_match=True),
                FieldSchema(name="desc_emb",
                            dtype=DataType.FLOAT_VECTOR,
                            dim=len(nodes_df_.iloc[0]['desc_emb'].to_arrow().to_pylist()[0])),
                FieldSchema(name="feat",
                            dtype=DataType.VARCHAR,
                            max_length=40960,
                            enable_analyzer=True,
                            enable_match=True),
                FieldSchema(name="feat_emb",
                            dtype=DataType.FLOAT_VECTOR,
                            dim=len(nodes_df_.iloc[0]['feat_emb'].to_arrow().to_pylist()[0])),
            ]
            schema = CollectionSchema(fields=node_fields,
                                      description=f"schema for collection {node_coll_name}")

            if not utility.has_collection(node_coll_name):
                collection = Collection(name=node_coll_name, schema=schema)
            else:
                collection = Collection(name=node_coll_name)

            # Create indexes
            collection.create_index(field_name="node_index",
                                    index_params={"index_type": "STL_SORT"},
                                    index_name="node_index_index")
            collection.create_index(field_name="node_name",
                                    index_params={"index_type": "INVERTED"},
                                    index_name="node_name_index")
            collection.create_index(field_name="node_type",
                                    index_params={"index_type": "INVERTED"},
                                    index_name="node_type_index")
            collection.create_index(field_name="desc",
                                    index_params={"index_type": "INVERTED"},
                                    index_name="desc_index")
            collection.create_index(field_name="desc_emb",
                                    index_params={"index_type": "GPU_CAGRA",
                                                  "metric_type": "IP"},
                                    index_name="desc_emb_index")
            collection.create_index(field_name="feat_emb",
                                    index_params={"index_type": "GPU_CAGRA",
                                                  "metric_type": "IP"},
                                    index_name="feat_emb_index")

            # Prepare data
            desc_emb_norm = cp.asarray(nodes_df_["desc_emb"].list.leaves).astype(cp.float32).\
                reshape(nodes_df_.shape[0], -1)
            desc_emb_norm = self.normalize_matrix(desc_emb_norm, axis=1)
            feat_emb_norm = cp.asarray(nodes_df_["feat_emb"].list.leaves).astype(cp.float32).\
                reshape(nodes_df_.shape[0], -1)
            feat_emb_norm = self.normalize_matrix(feat_emb_norm, axis=1)
            data = [
                nodes_df_["node_index"].to_arrow().to_pylist(),
                nodes_df_["node_id"].to_arrow().to_pylist(),
                nodes_df_["node_name"].to_arrow().to_pylist(),
                nodes_df_["node_type"].to_arrow().to_pylist(),
                nodes_df_["desc"].to_arrow().to_pylist(),
                desc_emb_norm.tolist(), # Use normalized embeddings
                nodes_df_["feat"].to_arrow().to_pylist(),
                feat_emb_norm.tolist(), # Use normalized embeddings
            ]

            # Insert data in batches
            total_rows = len(data[0])
            for i in range(0, total_rows, self.batch_size):
                batch = [col[i:i + self.batch_size] for col in data]
                collection.insert(batch)

            collection.flush()
            logger.info("Collection %s created with %d entities",
                        node_coll_name, collection.num_entities)

    def create_edges_collection(self,
                                edges_enrichment_df: cudf.DataFrame,
                                edges_embedding_df: List[cudf.DataFrame]):
        """Create and populate the edges collection."""
        logger.info("Creating edges collection...")

        edge_coll_name = f"{self.milvus_database}_edges"

        edge_fields = [
            FieldSchema(name="triplet_index",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=False),
            FieldSchema(name="head_id",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="head_index",
                        dtype=DataType.INT64),
            FieldSchema(name="tail_id",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="tail_index",
                        dtype=DataType.INT64),
            FieldSchema(name="edge_type",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="display_relation",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="feat",
                        dtype=DataType.VARCHAR,
                        max_length=40960),
            FieldSchema(name="feat_emb",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=len(edges_embedding_df[0].loc[0, 'edge_emb'])),
        ]
        edge_schema = CollectionSchema(fields=edge_fields,
                                       description="Schema for edges collection")

        if not utility.has_collection(edge_coll_name):
            collection = Collection(name=edge_coll_name, schema=edge_schema)
        else:
            collection = Collection(name=edge_coll_name)

        # Create indexes
        collection.create_index(field_name="triplet_index",
                                index_params={"index_type": "STL_SORT"},
                                index_name="triplet_index_index")
        collection.create_index(field_name="head_index",
                                index_params={"index_type": "STL_SORT"},
                                index_name="head_index_index")
        collection.create_index(field_name="tail_index",
                                index_params={"index_type": "STL_SORT"},
                                index_name="tail_index_index")
        collection.create_index(field_name="feat_emb",
                                index_params={"index_type": "GPU_CAGRA",
                                              "metric_type": "IP"},
                                index_name="feat_emb_index")

        # Iterate over chunked edges embedding df
        for edges_df in tqdm(edges_embedding_df):
            # Merge enrichment with embedding
            merged_edges_df = edges_enrichment_df.merge(
                edges_df[["triplet_index", "edge_emb"]],
                on="triplet_index",
                how="inner"
            )

            # Prepare data
            edge_emb_cp = cp.asarray(merged_edges_df["edge_emb"].list.leaves).astype(cp.float32).\
                reshape(merged_edges_df.shape[0], -1)
            edge_emb_norm = self.normalize_matrix(edge_emb_cp, axis=1)
            data = [
                merged_edges_df["triplet_index"].to_arrow().to_pylist(),
                merged_edges_df["head_id"].to_arrow().to_pylist(),
                merged_edges_df["head_index"].to_arrow().to_pylist(),
                merged_edges_df["tail_id"].to_arrow().to_pylist(),
                merged_edges_df["tail_index"].to_arrow().to_pylist(),
                merged_edges_df["edge_type_str"].to_arrow().to_pylist(),
                merged_edges_df["display_relation"].to_arrow().to_pylist(),
                merged_edges_df["feat"].to_arrow().to_pylist(),
                edge_emb_norm.tolist(), # Use normalized embeddings
            ]

            # Insert data in batches
            total = len(data[0])
            for i in tqdm(range(0, total, self.batch_size), desc="Inserting edges"):
                batch_data = [d[i:i+self.batch_size] for d in data]
                collection.insert(batch_data)

        collection.flush()
        logger.info("Edges collection created with %d entities", collection.num_entities)

    def run(self):
        """Main execution method."""
        try:
            logger.info("Starting Milvus data loading process...")

            # Connect to Milvus
            self.connect_to_milvus()

            # Load graph data
            graph = self.load_graph_data()

            # Prepare data
            logger.info("Data Preparation started...")
            # Get nodes enrichment and embedding dataframes
            nodes_enrichment_df = graph['nodes']['enrichment']
            nodes_embedding_df = graph['nodes']['embedding']

            # Get edges enrichment and embedding dataframes
            edges_enrichment_df = graph['edges']['enrichment']
            # !!consisted of a list of dataframes!!
            edges_embedding_df = graph['edges']['embedding']

            # For nodes, we can directly merge enrichment and embedding
            # Merge nodes enrichment and embedding dataframes
            merged_nodes_df = nodes_enrichment_df.merge(
                nodes_embedding_df[["node_id", "desc_emb", "feat_emb"]],
                on="node_id",
                how="left"
            )

            # Create collections and load data
            self.create_nodes_collection(merged_nodes_df)
            self.create_node_type_collections(merged_nodes_df)
            self.create_edges_collection(edges_enrichment_df,
                                         edges_embedding_df)

            # List all collections for verification
            logger.info("Data loading completed successfully!")
            logger.info("Created collections:")
            for coll in utility.list_collections():
                collection = Collection(name=coll)
                logger.info("  %s: %d entities", coll, collection.num_entities)

        except Exception as e:
            logger.error("Error during data loading: %s", str(e))
            raise


def main():
    """Main function to run the data loader."""
    # Resolve the fallback data path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "tests/files/biobridge_multimodal/")

    # Configuration
    config = {
        'milvus_host': os.getenv('MILVUS_HOST', 'localhost'),
        'milvus_port': os.getenv('MILVUS_PORT', '19530'),
        'milvus_user': os.getenv('MILVUS_USER', 'root'),
        'milvus_password': os.getenv('MILVUS_PASSWORD', 'Milvus'),
        'milvus_database': os.getenv('MILVUS_DATABASE', 't2kg_primekg'),
        'data_dir': os.getenv('DATA_DIR', default_data_dir),
        'batch_size': int(os.getenv('BATCH_SIZE', '500')),
        'chunk_size': int(os.getenv('CHUNK_SIZE', '5')),
    }

    # Print configuration for debugging
    print("[DATA LOADER] Configuration:")
    for key, value in config.items():
        print(f"[DATA LOADER]   {key}: {value}")

    # Create and run data loader
    loader = MilvusDataLoader(config)
    loader.run()


if __name__ == "__main__":
    main()
