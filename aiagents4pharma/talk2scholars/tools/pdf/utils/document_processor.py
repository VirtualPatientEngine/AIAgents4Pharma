"""
Document processing utilities for loading and splitting PDFs.
"""

import logging
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

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
