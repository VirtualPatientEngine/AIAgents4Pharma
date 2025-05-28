"""
NVIDIA NIM Reranker Utility
"""

import logging
import os


from typing import Any, List

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def rank_papers_by_query(self, query: str, config: Any, top_k: int = 5) -> List[str]:
    """
    Rank papers by relevance to the query using NVIDIA's off-the-shelf re-ranker.

    This function aggregates all chunks per paper, ranks them using the NVIDIA model,
    and returns the top-k papers.

    Args:
        query (str): The query string.
        config (Any): Configuration containing reranker settings (model, api_key).
        top_k (int): Number of top papers to return.

    Returns:
        List of tuples (paper_id, dummy_score) sorted by relevance.
    """

    logger.info("Starting NVIDIA re-ranker for query: '%s' with top_k=%d", query, top_k)
    # Aggregate all document chunks for each paper
    paper_texts = {}
    for doc in self.documents.values():
        paper_id = doc.metadata["paper_id"]
        paper_texts.setdefault(paper_id, []).append(doc.page_content)

    aggregated_documents = []
    for paper_id, texts in paper_texts.items():
        aggregated_text = " ".join(texts)
        aggregated_documents.append(
            Document(page_content=aggregated_text, metadata={"paper_id": paper_id})
        )

    logger.info(
        "Aggregated %d papers into %d documents for reranking",
        len(paper_texts),
        len(aggregated_documents),
    )
    # Instantiate the NVIDIA re-ranker client using provided config
    # Use NVIDIA API key from Hydra configuration (expected to be resolved via oc.env)
    api_key = config.reranker.api_key
    if not api_key:
        logger.error("No NVIDIA API key found in configuration for reranking")
        raise ValueError("Configuration 'reranker.api_key' must be set for reranking")
    logger.info("Using NVIDIA API key from configuration for reranking")
    # Truncate long inputs at the model-end to avoid exceeding max token size
    logger.info("Setting NVIDIA reranker truncate mode to END to limit input length")
    reranker = NVIDIARerank(
        model=config.reranker.model,
        api_key=api_key,
        truncate="END",
    )

    # Get the ranked list of documents based on the query
    response = reranker.compress_documents(query=query, documents=aggregated_documents)
    logger.info("Received %d documents from NVIDIA reranker", len(response))

    ranked_papers = [doc.metadata["paper_id"] for doc in response[:top_k]]
    logger.info("Top %d papers after reranking: %s", top_k, ranked_papers)
    return ranked_papers
