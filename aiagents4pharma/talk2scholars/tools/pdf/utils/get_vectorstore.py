"""
Create or retrieve a Vectorstore instance for PDF RAG.
"""

import logging
import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from .vector_store import Vectorstore

logger = logging.getLogger(__name__)

# Global cache for Vectorstore instances
_vectorstore_cache = {}
_cache_lock = threading.Lock()


def get_vectorstore(
    embedding_model: Embeddings, config: Any, force_new: bool = False
) -> "Vectorstore":
    """
    Factory function to get or create a Vectorstore instance.
    Ensures the same instance is reused across the application.

    Args:
        embedding_model: The embedding model to use
        config: Configuration object
        force_new: Force creation of a new instance

    Returns:
        Vectorstore instance
    """
    collection_name = config.milvus.collection_name if config else "pdf_rag_documents"

    with _cache_lock:
        if force_new and collection_name in _vectorstore_cache:
            del _vectorstore_cache[collection_name]
            logger.info("Forced new Vectorstore instance for collection: %s", collection_name)

        if collection_name not in _vectorstore_cache:
            logger.info("Creating new Vectorstore instance for collection: %s", collection_name)
            _vectorstore_cache[collection_name] = Vectorstore(
                embedding_model=embedding_model, config=config
            )
        else:
            logger.info(
                "Reusing existing Vectorstore instance for collection: %s",
                collection_name,
            )
            # Update embedding model if different
            existing = _vectorstore_cache[collection_name]
            if existing.embedding_model != embedding_model:
                logger.warning("Embedding model changed, updating existing instance")
                existing.embedding_model = embedding_model
                existing.vector_store.embedding_function = embedding_model

        return _vectorstore_cache[collection_name]
