"""
Retrieve relevant chunks from a vector store using MMR (Maximal Marginal Relevance).
"""

import logging
import os
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import maximal_marginal_relevance


# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def retrieve_relevant_chunks(
    self,
    query: str,
    paper_ids: Optional[List[str]] = None,
    top_k: int = 25,
    mmr_diversity: float = 1.00,
) -> List[Document]:
    """
    Retrieve the most relevant chunks for a query using maximal marginal relevance.

    Args:
        query: Query string
        paper_ids: Optional list of paper IDs to filter by
        top_k: Number of chunks to retrieve
        mmr_diversity: Diversity parameter for MMR (higher = more diverse)

    Returns:
        List of document chunks
    """
    if not self.vector_store:
        logger.error("Failed to build vector store")
        return []

    if paper_ids:
        logger.info("Filtering retrieval to papers: %s", paper_ids)

    # Step 1: Embed the query
    logger.info("Embedding query using model: %s", type(self.embedding_model).__name__)
    query_embedding = np.array(self.embedding_model.embed_query(query))

    # Step 2: Filter relevant documents
    all_docs = [
        doc
        for doc in self.documents.values()
        if not paper_ids or doc.metadata["paper_id"] in paper_ids
    ]

    if not all_docs:
        logger.warning("No documents found after filtering by paper_ids.")
        return []

    # Step 3: Retrieve or compute embeddings for all documents using cache
    logger.info("Retrieving embeddings for %d chunks...", len(all_docs))
    all_embeddings = []
    for doc in all_docs:
        doc_id = f"{doc.metadata['paper_id']}_{doc.metadata['chunk_id']}"
        if doc_id not in self.embeddings:
            logger.info("Embedding missing chunk %s", doc_id)
            emb = self.embedding_model.embed_documents([doc.page_content])[0]
            self.embeddings[doc_id] = emb
        all_embeddings.append(self.embeddings[doc_id])

    # Step 4: Apply MMR
    mmr_indices = maximal_marginal_relevance(
        query_embedding,
        all_embeddings,
        k=top_k,
        lambda_mult=mmr_diversity,
    )

    results = [all_docs[i] for i in mmr_indices]
    logger.info("Retrieved %d chunks using MMR", len(results))
    return results
