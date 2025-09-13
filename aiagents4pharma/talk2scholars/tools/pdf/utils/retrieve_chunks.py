"""
Retrieve relevant chunks from a Milvus vector store using MMR (Maximal Marginal Relevance).
Follows traditional RAG pipeline - retrieve first, then rerank.
With automatic GPU/CPU search parameter optimization.
"""

import logging
import os

from langchain_core.documents import Document

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    paper_ids: list[str] | None = None,
    top_k: int = 100,  # Increased default to cast wider net before reranking
    mmr_diversity: float = 0.8,  # Slightly reduced for better diversity
) -> list[Document]:
    """
    Retrieve the most relevant chunks for a query using maximal marginal relevance.
    Automatically uses GPU-optimized search parameters if GPU is available.

    In the traditional RAG pipeline, this should retrieve chunks from ALL available papers,
    not just pre-selected ones. The reranker will then select the best chunks.

    Args:
        vector_store: The Milvus vector store instance
        query: Query string
        paper_ids: Optional list of paper IDs to filter by (default: None - search all papers)
        top_k: Number of chunks to retrieve (default: 100 for reranking pipeline)
        mmr_diversity: Diversity parameter for MMR (0=max diversity, 1=max relevance)

    Returns:
        List of document chunks
    """
    if not vector_store:
        logger.error("Vector store is not initialized")
        return []

    # Check if vector store has GPU capabilities
    has_gpu = getattr(vector_store, "has_gpu", False)
    search_mode = "GPU-accelerated" if has_gpu else "CPU"

    # Prepare filter for paper_ids if provided
    filter_dict = None
    if paper_ids:
        logger.warning(
            "Paper IDs filter provided. Traditional RAG pipeline typically"
            "retrieves from ALL papers first. "
            "Consider removing paper_ids filter for better results."
        )
        logger.info("Filtering retrieval to papers: %s", paper_ids)
        filter_dict = {"paper_id": paper_ids}
    else:
        logger.info(
            "Retrieving chunks from ALL papers (traditional RAG approach) using %s search",
            search_mode,
        )

    # Use Milvus's built-in MMR search with optimized parameters
    logger.info(
        "Performing %s MMR search with query: '%s', k=%d, diversity=%.2f",
        search_mode,
        query[:50] + "..." if len(query) > 50 else query,
        top_k,
        mmr_diversity,
    )

    # Fetch more candidates for better MMR results
    # Adjust fetch_k based on available hardware
    if has_gpu:
        # GPU can handle larger candidate sets efficiently
        fetch_k = min(top_k * 6, 800)  # Increased for GPU
        logger.debug("Using GPU-optimized fetch_k: %d", fetch_k)
    else:
        # CPU - more conservative to avoid performance issues
        fetch_k = min(top_k * 4, 500)  # Original conservative approach
        logger.debug("Using CPU-optimized fetch_k: %d", fetch_k)

    # Get search parameters from vector store if available
    search_params = getattr(vector_store, "search_params", None)

    if search_params:
        logger.debug("Using hardware-optimized search parameters: %s", search_params)
    else:
        logger.debug("Using default search parameters (no hardware optimization)")

    # Perform MMR search - let the vector store handle search_params internally
    # Don't pass search_params explicitly to avoid conflicts
    results = vector_store.max_marginal_relevance_search(
        query=query,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=mmr_diversity,
        filter=filter_dict,
    )

    logger.info("Retrieved %d chunks using %s MMR from Milvus", len(results), search_mode)

    # Log some details about retrieved chunks for debugging
    if results and logger.isEnabledFor(logging.DEBUG):
        paper_counts = {}
        for doc in results:
            paper_id = doc.metadata.get("paper_id", "unknown")
            paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1

        logger.debug(
            "%s retrieval - chunks per paper: %s",
            search_mode,
            dict(sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        )
        logger.debug(
            "%s retrieval - total papers represented: %d",
            search_mode,
            len(paper_counts),
        )

    return results


def retrieve_relevant_chunks_with_scores(
    vector_store,
    query: str,
    paper_ids: list[str] | None = None,
    top_k: int = 100,
    score_threshold: float = 0.0,
) -> list[tuple[Document, float]]:
    """
    Retrieve chunks with similarity scores, optimized for GPU/CPU.

    Args:
        vector_store: The Milvus vector store instance
        query: Query string
        paper_ids: Optional list of paper IDs to filter by
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity score threshold

    Returns:
        List of (document, score) tuples
    """
    if not vector_store:
        logger.error("Vector store is not initialized")
        return []

    has_gpu = getattr(vector_store, "has_gpu", False)
    search_mode = "GPU-accelerated" if has_gpu else "CPU"

    # Prepare filter
    filter_dict = None
    if paper_ids:
        filter_dict = {"paper_id": paper_ids}

    logger.info(
        "Performing %s similarity search with scores: query='%s', k=%d, threshold=%.3f",
        search_mode,
        query[:50] + "..." if len(query) > 50 else query,
        top_k,
        score_threshold,
    )

    # Check hardware optimization status instead of unused search_params
    has_optimization = hasattr(vector_store, "has_gpu") and vector_store.has_gpu

    if has_optimization:
        logger.debug("GPU-accelerated similarity search enabled")
    else:
        logger.debug("Standard CPU similarity search")

    if hasattr(vector_store, "similarity_search_with_score"):
        # Don't pass search_params to avoid conflicts
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict,
        )

        # Filter by score threshold
        filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

        logger.info(
            "%s search with scores retrieved %d/%d chunks above threshold %.3f",
            search_mode,
            len(filtered_results),
            len(results),
            score_threshold,
        )

        return filtered_results

    raise NotImplementedError("Vector store does not support similarity_search_with_score")
