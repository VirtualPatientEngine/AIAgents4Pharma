"""
Paper loading utilities for managing PDF documents in vector store.
"""

import logging
from typing import Any

from .batch_processor import add_papers_batch

logger = logging.getLogger(__name__)


def load_all_papers(
    vector_store: Any,  # The Vectorstore instance
    articles: dict[str, Any],
    call_id: str,
    config: Any,
    has_gpu: bool,
) -> None:
    """
    Ensure all papers from article_data are loaded into the Milvus vector store.
    Optimized for GPU/CPU processing.

    Args:
        vector_store: The Vectorstore instance
        articles: Dictionary of article data
        call_id: Call identifier for logging
        config: Configuration object
        has_gpu: Whether GPU is available
    """
    papers_to_load = []
    skipped_papers = []
    already_loaded = []

    # Check which papers need to be loaded
    for pid, article_info in articles.items():
        if pid not in vector_store.loaded_papers:
            pdf_url = article_info.get("pdf_url")
            if pdf_url:
                # Prepare tuple for batch loading
                papers_to_load.append((pid, pdf_url, article_info))
            else:
                skipped_papers.append(pid)
        else:
            already_loaded.append(pid)

    # Log summary of papers status with hardware info
    hardware_info = f" (GPU acceleration: {'enabled' if has_gpu else 'disabled'})"
    logger.info(
        "%s: Paper loading summary%s - Total: %d, Already loaded: %d, To load: %d, No PDF: %d",
        call_id,
        hardware_info,
        len(articles),
        len(already_loaded),
        len(papers_to_load),
        len(skipped_papers),
    )

    if skipped_papers:
        logger.warning(
            "%s: Skipping %d papers without PDF URLs: %s%s",
            call_id,
            len(skipped_papers),
            skipped_papers[:5],  # Show first 5
            "..." if len(skipped_papers) > 5 else "",
        )

    if not papers_to_load:
        logger.info("%s: All papers with PDFs are already loaded in Milvus", call_id)
        return

    # Use batch loading with parallel processing for ALL papers at once
    # Adjust parameters based on hardware capabilities
    if has_gpu:
        # GPU can handle more parallel processing
        max_workers = min(12, max(4, len(papers_to_load)))  # More workers for GPU
        batch_size = config.get("embedding_batch_size", 2000)  # Larger batches for GPU
        logger.info(
            "%s: Using GPU-optimized loading parameters: %d workers, batch size %d",
            call_id,
            max_workers,
            batch_size,
        )
    else:
        # CPU - more conservative parameters
        max_workers = min(8, max(3, len(papers_to_load)))  # Conservative for CPU
        batch_size = config.get("embedding_batch_size", 1000)  # Smaller batches for CPU
        logger.info(
            "%s: Using CPU-optimized loading parameters: %d workers, batch size %d",
            call_id,
            max_workers,
            batch_size,
        )

    logger.info(
        "%s: Loading %d papers in ONE BATCH using %d parallel workers (batch size: %d, %s)",
        call_id,
        len(papers_to_load),
        max_workers,
        batch_size,
        "GPU accelerated" if has_gpu else "CPU processing",
    )

    # This should process ALL papers at once with hardware optimization
    add_papers_batch(
        papers_to_add=papers_to_load,
        vector_store=vector_store.vector_store,  # Pass the LangChain vector store
        loaded_papers=vector_store.loaded_papers,
        paper_metadata=vector_store.paper_metadata,
        documents=vector_store.documents,
        config=vector_store.config,
        metadata_fields=vector_store.metadata_fields,
        has_gpu=vector_store.has_gpu,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    logger.info(
        "%s: Successfully completed batch loading of all %d papers with %s",
        call_id,
        len(papers_to_load),
        "GPU acceleration" if has_gpu else "CPU processing",
    )
