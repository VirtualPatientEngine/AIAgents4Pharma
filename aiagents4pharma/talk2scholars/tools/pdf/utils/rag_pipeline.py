"""
RAG pipeline for retrieving and reranking chunks from a vector store.
"""

import logging
from typing import Any

# Import our GPU detection utility
from .nvidia_nim_reranker import rerank_chunks
from .retrieve_chunks import retrieve_relevant_chunks

logger = logging.getLogger(__name__)


def retrieve_and_rerank_chunks(
    vector_store: Any, query: str, config: Any, call_id: str, has_gpu: bool
) -> list[Any]:
    """
    Traditional RAG pipeline: retrieve chunks from all papers, then rerank.
    Optimized for GPU/CPU hardware.

    Args:
        vs: Vector store instance
        query: User query

    Returns:
        List of reranked chunks
    """
    hardware_mode = "GPU-accelerated" if has_gpu else "CPU-optimized"
    logger.info(
        "%s: Starting traditional RAG pipeline - retrieve then rerank (%s)",
        call_id,
        hardware_mode,
    )

    # Step 1: Retrieve chunks from ALL papers (cast wide net)
    # Adjust initial retrieval count based on hardware
    if has_gpu:
        # GPU can handle larger initial retrieval efficiently
        initial_chunks_count = config.get("initial_retrieval_k", 150)  # Increased for GPU
        mmr_diversity = config.get("mmr_diversity", 0.75)  # Slightly more diverse for larger sets
    else:
        # CPU - use conservative settings
        initial_chunks_count = config.get("initial_retrieval_k", 100)  # Original
        mmr_diversity = config.get("mmr_diversity", 0.8)  # Original

    logger.info(
        "%s: Step 1 - Retrieving top %d chunks from ALL papers (%s mode)",
        call_id,
        initial_chunks_count,
        hardware_mode,
    )

    retrieved_chunks = retrieve_relevant_chunks(
        vector_store,
        query=query,
        paper_ids=None,  # No filter - retrieve from all papers
        top_k=initial_chunks_count,
        mmr_diversity=mmr_diversity,
    )

    if not retrieved_chunks:
        logger.warning("%s: No chunks retrieved from vector store", call_id)
        return []

    logger.info(
        "%s: Retrieved %d chunks from %d unique papers using %s",
        call_id,
        len(retrieved_chunks),
        len({chunk.metadata.get("paper_id", "unknown") for chunk in retrieved_chunks}),
        hardware_mode,
    )

    # Step 2: Rerank the retrieved chunks
    final_chunk_count = config.top_k_chunks
    logger.info(
        "%s: Step 2 - Reranking %d chunks to get top %d",
        call_id,
        len(retrieved_chunks),
        final_chunk_count,
    )

    reranked_chunks = rerank_chunks(
        chunks=retrieved_chunks,
        query=query,
        config=config,
        top_k=final_chunk_count,
    )

    # Log final results with hardware info
    final_papers = len({chunk.metadata.get("paper_id", "unknown") for chunk in reranked_chunks})

    logger.info(
        "%s: Reranking complete using %s. Final %d chunks from %d unique papers",
        call_id,
        hardware_mode,
        len(reranked_chunks),
        final_papers,
    )

    # Log performance insights
    if len(retrieved_chunks) > 0:
        efficiency = len(reranked_chunks) / len(retrieved_chunks) * 100
        logger.debug(
            "%s: Pipeline efficiency: %.1f%% (%d final / %d initial chunks) - %s",
            call_id,
            efficiency,
            len(reranked_chunks),
            len(retrieved_chunks),
            hardware_mode,
        )

    return reranked_chunks
