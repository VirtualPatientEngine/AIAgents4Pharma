"""
Vector normalization utilities for GPU COSINE similarity support.
Since GPU indexes don't support COSINE distance, we normalize vectors
and use IP (Inner Product) distance instead.
"""

import logging

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def normalize_vector(vector: list[float] | np.ndarray) -> list[float]:
    """
    Normalize a single vector to unit length.

    Args:
        vector: Input vector as list or numpy array

    Returns:
        Normalized vector as list
    """
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        logger.warning("Zero vector encountered during normalization")
        return vector.tolist()

    normalized = vector / norm
    return normalized.tolist()


def normalize_vectors_batch(vectors: list[list[float]]) -> list[list[float]]:
    """
    Normalize a batch of vectors to unit length.

    Args:
        vectors: List of vectors

    Returns:
        List of normalized vectors
    """
    if not vectors:
        return vectors

    # Convert to numpy array for efficient computation
    vectors_array = np.asarray(vectors, dtype=np.float32)

    # Calculate norms for each vector
    norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)

    # Handle zero vectors
    zero_mask = norms.flatten() == 0
    if np.any(zero_mask):
        logger.warning("Found %d zero vectors during batch normalization", np.sum(zero_mask))
        norms[zero_mask] = 1.0  # Avoid division by zero

    # Normalize
    normalized = vectors_array / norms

    return normalized.tolist()


class NormalizingEmbeddings(Embeddings):
    """
    Wrapper around an embedding model that automatically normalizes outputs.
    This is needed for GPU indexes when using COSINE similarity.
    """

    def __init__(self, embedding_model: Embeddings, normalize_for_gpu: bool = True):
        """
        Initialize the normalizing wrapper.

        Args:
            embedding_model: The underlying embedding model
            normalize_for_gpu: Whether to normalize embeddings (for GPU compatibility)
        """
        self.embedding_model = embedding_model
        self.normalize_for_gpu = normalize_for_gpu

        if normalize_for_gpu:
            logger.info("Embedding model wrapped with normalization for GPU compatibility")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents and optionally normalize."""
        embeddings = self.embedding_model.embed_documents(texts)

        if self.normalize_for_gpu:
            embeddings = normalize_vectors_batch(embeddings)
            logger.debug("Normalized %d document embeddings for GPU", len(embeddings))

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query and optionally normalize."""
        embedding = self.embedding_model.embed_query(text)

        if self.normalize_for_gpu:
            embedding = normalize_vector(embedding)
            logger.debug("Normalized query embedding for GPU")

        return embedding

    def __getattr__(self, name):
        """Delegate other attributes to the underlying model."""
        return getattr(self.embedding_model, name)


def should_normalize_vectors(has_gpu: bool, use_cosine: bool) -> bool:
    """
    Determine if vectors should be normalized based on hardware and similarity metric.

    Args:
        has_gpu: Whether GPU is being used
        use_cosine: Whether COSINE similarity is desired

    Returns:
        True if vectors should be normalized
    """
    needs_normalization = has_gpu and use_cosine

    if needs_normalization:
        logger.info("Vector normalization ENABLED: GPU detected with COSINE similarity request")
    else:
        logger.info("Vector normalization DISABLED: GPU=%s, COSINE=%s", has_gpu, use_cosine)

    return needs_normalization


def wrap_embedding_model_if_needed(
    embedding_model: Embeddings, has_gpu: bool, use_cosine: bool = True
) -> Embeddings:
    """
    Wrap embedding model with normalization if needed for GPU compatibility.

    Args:
        embedding_model: Original embedding model
        has_gpu: Whether GPU is being used
        use_cosine: Whether COSINE similarity is desired

    Returns:
        Original or wrapped embedding model
    """
    if should_normalize_vectors(has_gpu, use_cosine):
        return NormalizingEmbeddings(embedding_model, normalize_for_gpu=True)

    return embedding_model
