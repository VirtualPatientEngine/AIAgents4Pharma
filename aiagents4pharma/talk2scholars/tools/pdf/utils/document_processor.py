"""
Document processing utilities for loading and splitting PDFs.
"""

import logging
import os
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from . import multimodal_processor as mp


logger = logging.getLogger(__name__)


def load_and_split_pdf(
    paper_id: str,
    pdf_url: str,
    paper_metadata: dict[str, Any],
    config: Any,
    **kwargs: Any,
) -> list[Document]:
    """
    Load a PDF and split it into chunks.

    Args:
        paper_id: Unique identifier for the paper.
        pdf_url: URL to the PDF.
        paper_metadata: Metadata about the paper (e.g. Title, Authors, etc.).
        config: Configuration object with `chunk_size` and `chunk_overlap` attributes.
        metadata_fields: List of additional metadata keys to propagate into each
        chunk (passed via kwargs).
        documents_dict: Dictionary where split chunks will also be stored under keys
            of the form "{paper_id}_{chunk_index}" (passed via kwargs).

    Returns:
        A list of Document chunks, each with updated metadata.
    """
    metadata_fields: list[str] = kwargs["metadata_fields"]
    documents_dict: dict[str, Document] = kwargs["documents_dict"]

    logger.info("Loading PDF for paper %s from %s", paper_id, pdf_url)

    # Load pages
    documents = PyPDFLoader(pdf_url).load()
    logger.info("Loaded %d pages from paper %s", len(documents), paper_id)

    if config is None:
        raise ValueError("Configuration is required for text splitting in Vectorstore.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split into chunks
    chunks = splitter.split_documents(documents)
    logger.info("Split paper %s into %d chunks", paper_id, len(chunks))

    # run multimodal processor at batch level
    multimodal_texts = []
    try:
        # Step 1: Convert PDF to base64 images
        base64_pages = mp.pdf_to_base64_compressed(pdf_url)

        # Step 2: Detect page elements (charts, tables, infographics)
        detected = mp.detect_page_elements(base64_pages)
        categorized = mp.categorize_page_elements(detected)

        # Step 3: Crop & process each element
        cropped = mp.crop_categorized_elements(categorized, base64_pages)
        processed = mp.process_all(cropped)

        # Step 4: Collect OCR/text results
        ocr_results = mp.collect_ocr_results(processed)
        lines = mp.extract_text_lines(ocr_results)

        # Flatten into text for RAG augmentation
        multimodal_texts.extend([line["text"] for line in lines])

        if chunks:
            chunks[0].metadata["multimodal_results"] = multimodal_texts
            logger.info("Attached multimodal results to first chunk of paper %s", paper_id)
    except Exception as e:
        logger.error(f"Error processing multimodal data for paper {paper_id}: {e}")
    # Attach metadata & populate documents_dict
    for i, chunk in enumerate(chunks):
        chunk_id = f"{paper_id}_{i}"
        chunk.metadata.update(
            {
                "paper_id": paper_id,
                "title": paper_metadata.get("Title", "Unknown"),
                "chunk_id": i,
                "page": chunk.metadata.get("page", 0),
                "source": pdf_url,
            }
        )
        for field in metadata_fields:
            if field in paper_metadata and field not in chunk.metadata:
                chunk.metadata[field] = paper_metadata[field]
        documents_dict[chunk_id] = chunk

    return chunks
