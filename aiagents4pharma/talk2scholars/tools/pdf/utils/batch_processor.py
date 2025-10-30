"""
Batch processing utilities for adding multiple papers to vector store.
"""

import concurrent.futures
import logging
import os
import time
from typing import Any, Dict, List, Set, Tuple
from langchain_core.documents import Document

from .document_processor import load_and_split_pdf
from . import multimodal_processor as mp


logger = logging.getLogger(__name__)


def add_papers_batch(
    papers_to_add: List[Tuple[str, str, Dict[str, Any]]],
    vector_store: Any,
    loaded_papers: Set[str],
    paper_metadata: Dict[str, Dict[str, Any]],
    documents: Dict[str, Document],
    **kwargs: Any,
) -> None:
    """
    Add multiple papers to the document store in parallel with batch embedding.

    Args:
        papers_to_add: List of tuples (paper_id, pdf_url, paper_metadata).
        vector_store: The LangChain Milvus vector store instance.
        loaded_papers: Set to track which papers are already loaded.
        paper_metadata: Dict to store paper metadata after load.
        documents: Dict to store document chunks.
        config:           (via kwargs) Configuration object.
        metadata_fields:  (via kwargs) List of metadata fields to include.
        has_gpu:          (via kwargs) Whether GPU is available.
        max_workers:      (via kwargs) Max PDF‐loading threads (default 5).
        batch_size:       (via kwargs) Embedding batch size (default 100).
    """
    cfg = kwargs

    if not papers_to_add:
        logger.info("No papers to add")
        return

    to_process = [
        (pid, url, md) for pid, url, md in papers_to_add if pid not in loaded_papers
    ]
    if not to_process:
        logger.info("Skipping %d already-loaded papers", len(papers_to_add))
        logger.info("All %d papers are already loaded", len(papers_to_add))
        return

    logger.info(
        "Starting PARALLEL batch processing of %d papers with %d workers (%s)",
        len(to_process),
        cfg.get("max_workers", 5),
        "GPU acceleration" if cfg["has_gpu"] else "CPU processing",
    )
    chunks, ids, success, multimodal_results = _parallel_load_and_split(
        to_process,
        cfg["config"],
        cfg["metadata_fields"],
        documents,
        cfg.get("max_workers", 5),
    )

    if not chunks:
        logger.warning("No chunks to add to vector store")
        return

    for pid, _, md in to_process:
        if pid in success:
            md["multimodal_results"] = multimodal_results.get(pid, {})
            paper_metadata[pid] = md

    try:
        _batch_embed(
            chunks,
            ids,
            vector_store,
            cfg.get("batch_size", 100),
            cfg["has_gpu"],
        )
    except Exception:
        logger.error("Failed to add chunks to Milvus", exc_info=True)
        raise

    # finally mark papers as loaded
    loaded_papers.update(success)


def _parallel_load_and_split(
    papers: List[Tuple[str, str, Dict[str, Any]]],
    config: Any,
    metadata_fields: List[str],
    documents: Dict[str, Document],
    max_workers: int,
) -> Tuple[List[Document], List[str], List[str], Dict[str, Any]]:
    """Load & split PDFs in parallel, and collect multimodal results per paper."""
    all_chunks: List[Document] = []
    all_ids: List[str] = []
    success: List[str] = []
    text_lines: Dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                load_and_split_pdf,
                pid,
                url,
                md,
                config,
                metadata_fields=metadata_fields,
                documents_dict=documents,
            ): (pid, url)
            for pid, url, md in papers
        }
        logger.info("Submitted %d PDF loading tasks", len(futures))

        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            pid, url = futures[fut]
            chunks = fut.result()
            ids = [f"{pid}_{i}" for i in range(len(chunks))]

            all_chunks.extend(chunks)
            all_ids.extend(ids)
            success.append(pid)

            # run multimodal processor at batch level
            multimodal_texts = []
            try:
                # Step 1: Convert PDF to base64 images
                base64_pages = mp.pdf_to_base64_compressed(url)

                # Step 2: Detect page elements (charts, tables, infographics)
                detected = mp.detect_page_elements(base64_pages)
                categorized = mp.categorize_page_elements(detected)

                # Step 3: Crop & process each element
                cropped = mp.crop_categorized_elements(categorized, base64_pages)
                processed = mp.process_all(cropped)

                # Step 4: Collect OCR/text results
                ocr_results = mp.collect_ocr_results(processed)
                lines = mp.extract_text_lines(ocr_results)
                text_lines[pid] = lines

                # Flatten into text for RAG augmentation
                multimodal_texts.extend([line["text"] for line in lines])

                # Convert multimodal text into Document objects
                multi_docs = [
                    Document(
                        page_content=line["text"],
                        metadata={
                            "paper_id": pid,
                            "source": "multimodal",
                            "page": line["page"] 
                        }
                    )
                    for line in lines
                ]

                multi_ids = [f"{pid}_multimodal_{i}" for i in range(len(multi_docs))]
                all_chunks.extend(multi_docs)
                all_ids.extend(multi_ids)
                # text_lines[pid] = multimodal_texts
            except Exception as e:
                logger.error("Multimodal processing failed for %s: %s", pid, e)

            logger.info(
                "Progress: %d/%d - Loaded paper %s (%d chunks)",
                idx,
                len(papers),
                pid,
                len(chunks),
            )

    return all_chunks, all_ids, success, text_lines

def _batch_embed(
    chunks: List[Document],
    ids: List[str],
    store: Any,
    batch_size: int,
    has_gpu: bool,
) -> None:
    """Embed chunks in batches and verify insertion exactly as before."""
    start = time.time()
    n = len(chunks)
    logger.info(
        "Starting BATCH EMBEDDING of %d chunks in batches of %d (%s)",
        n,
        batch_size,
        "GPU" if has_gpu else "CPU",
    )

    for batch_num, start_idx in enumerate(range(0, n, batch_size), start=1):
        end_idx = min(start_idx + batch_size, n)
        logger.info(
            "Embedding batch %d/%d (chunks %d-%d of %d) - %s",
            batch_num,
            (n + batch_size - 1) // batch_size,
            start_idx + 1,
            end_idx,
            n,
            "GPU" if has_gpu else "CPU",
        )

        store.add_documents(
            documents=chunks[start_idx:end_idx],
            ids=ids[start_idx:end_idx],
        )

        # Post-insert verification
        col = store.col
        col.flush()
        count = col.num_entities
        logger.info(
            "Post-insert batch %d: collection has %d entities",
            batch_num,
            count,
        )
        if count:
            logger.info(
                "Sample paper IDs: %s",
                [
                    r.get("paper_id", "unknown")
                    for r in col.query(expr="", output_fields=["paper_id"], limit=3)
                ],
            )

        logger.info("Successfully stored batch %d", batch_num)

    elapsed = time.time() - start
    logger.info(
        "BATCH EMBEDDING COMPLETE: %d chunks in %.2f seconds (%.2f chunks/sec)",
        n,
        elapsed,
        n / elapsed if elapsed > 0 else 0,
    )
