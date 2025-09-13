"""
NVIDIA NIM Reranker Utility for Milvus Integration
Rerank chunks instead of papers following traditional RAG pipeline
"""

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def rerank_chunks(
    chunks: list[Document], query: str, config: Any, top_k: int = 25
) -> list[Document]:
    """
    Rerank chunks by relevance to the query using NVIDIA's reranker.

    This follows the traditional RAG pipeline: first retrieve chunks, then rerank them.

    Args:
        chunks (List[Document]): List of chunks to rerank
        query (str): The query string
        config (Any): Configuration containing reranker settings
        top_k (int): Number of top chunks to return after reranking

    Returns:
        List[Document]: Reranked chunks (top_k most relevant)
    """
    logger.info(
        "Starting NVIDIA chunk reranker for query: '%s' with %d chunks, top_k=%d",
        query[:50] + "..." if len(query) > 50 else query,
        len(chunks),
        top_k,
    )

    # If we have fewer chunks than top_k, just return all
    if len(chunks) <= top_k:
        logger.info(
            "Number of chunks (%d) <= top_k (%d), returning all chunks without reranking",
            len(chunks),
            top_k,
        )
        return chunks

    # Get API key from config
    api_key = config.reranker.api_key
    if not api_key:
        logger.error("No NVIDIA API key found in configuration for reranking")
        raise ValueError("Configuration 'reranker.api_key' must be set for reranking")

    logger.info("Using NVIDIA reranker model: %s", config.reranker.model)

    # Initialize reranker with truncation to handle long chunks
    reranker = NVIDIARerank(
        model=config.reranker.model,
        api_key=api_key,
        truncate="END",  # Truncate at the end if too long
    )

    # Log chunk metadata for debugging
    logger.debug(
        "Reranking chunks from papers: %s",
        list({chunk.metadata.get("paper_id", "unknown") for chunk in chunks})[:5],
    )

    # Rerank the chunks
    logger.info("Calling NVIDIA reranker API with %d chunks...", len(chunks))
    reranked_chunks = reranker.compress_documents(query=query, documents=chunks)

    for i, doc in enumerate(reranked_chunks[:top_k]):
        score = doc.metadata.get("relevance_score", "N/A")
        source = doc.metadata.get("paper_id", "unknown")
        logger.info("Rank %d | Score: %.4f | Source: %s", i + 1, score, source)

    logger.info(
        "Successfully reranked chunks. Returning top %d chunks",
        min(top_k, len(reranked_chunks)),
    )

    # Log which papers the top chunks come from
    if reranked_chunks and logger.isEnabledFor(logging.DEBUG):
        top_papers = {}
        for chunk in reranked_chunks[:top_k]:
            paper_id = chunk.metadata.get("paper_id", "unknown")
            top_papers[paper_id] = top_papers.get(paper_id, 0) + 1
        logger.debug("Top %d chunks distribution by paper: %s", top_k, top_papers)

    # Return only top_k chunks (convert to list to match return type)
    return list(reranked_chunks[:top_k])
