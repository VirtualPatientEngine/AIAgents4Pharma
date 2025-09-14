"""
Collection Manager for Milvus
"""

import logging
import os
import threading
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))

# Global cache for collections to avoid repeated creation checks
_collection_cache = {}
_cache_lock = threading.Lock()


def ensure_collection_exists(
    collection_name: str, config: Any, index_params: dict[str, Any], has_gpu: bool
) -> Collection:
    """Ensure the Milvus collection exists before trying to sync or add documents."""

    # Check cache first
    with _cache_lock:
        if collection_name in _collection_cache:
            logger.debug("Returning cached collection: %s", collection_name)
            return _collection_cache[collection_name]

    try:
        existing_collections = utility.list_collections()
        if collection_name not in existing_collections:
            logger.info(
                "Collection %s does not exist. Creating schema...",
                collection_name,
            )

            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=100,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=config.milvus.embedding_dim if config else 768,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="paper_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                ),
                FieldSchema(
                    name="title",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="chunk_id",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="page",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="RAG collection for embedded PDF chunks",
                enable_dynamic_field=True,
            )

            # Create collection
            collection = Collection(
                name=collection_name,
                schema=schema,
                using="default",
                shards_num=2,
            )
            logger.info("Created collection: %s", collection_name)

            # Create index on the embedding field with GPU/CPU optimization
            logger.info(
                "Creating %s index on 'embedding' field for collection: %s",
                index_params["index_type"],
                collection_name,
            )

            collection.create_index(field_name="embedding", index_params=index_params)

            index_type = index_params["index_type"]
            logger.info(
                "Successfully created %s index on 'embedding' field for collection: %s",
                index_type,
                collection_name,
            )

        else:
            logger.info("Collection %s already exists. Loading it.", collection_name)
            collection = Collection(name=collection_name, using="default")

        collection.load()

        def debug_collection_state(collection, collection_name):
            """Debug collection state for troubleshooting."""
            logger.info("=== DEBUG COLLECTION STATE ===")
            logger.info("Collection name: %s", collection_name)
            logger.info("Collection schema: %s", collection.schema)
            logger.info("Collection num_entities: %d", collection.num_entities)

            # Check if collection is actually loaded
            # logger.info("Is collection loaded: %s", collection.load)

            # Check available indexes
            indexes = collection.indexes
            logger.info("Collection indexes: %s", [idx.field_name for idx in indexes])

            # Try to get collection stats
            logger.info("Collection statistics: %s", collection.num_entities)

            logger.info("Active connections: %s", connections.list_connections())

            logger.info("=== END DEBUG ===")

        debug_collection_state(collection, collection_name)

        # Log collection statistics with GPU/CPU info
        num_entities = collection.num_entities
        gpu_info = " (GPU accelerated)" if has_gpu else " (CPU only)"
        logger.info(
            "Collection %s is loaded and ready with %d entities%s",
            collection_name,
            num_entities,
            gpu_info,
        )

        # Cache the collection
        with _cache_lock:
            _collection_cache[collection_name] = collection
            logger.debug("Cached collection: %s", collection_name)

        return collection  # Return the collection object

    except Exception as e:
        logger.error("Failed to ensure collection exists: %s", e, exc_info=True)
        raise
