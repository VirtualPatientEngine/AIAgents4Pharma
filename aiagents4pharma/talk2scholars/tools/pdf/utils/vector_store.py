"""
Vectorstore class for managing document embeddings and retrieval.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class Vectorstore:
    """
    A class for managing document embeddings and retrieval.
    Provides unified access to documents across multiple papers.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        metadata_fields: Optional[List[str]] = None,
        config: Any = None,
    ):
        """
        Initialize the document store.

        Args:
            embedding_model: The embedding model to use
            metadata_fields: Fields to include in document metadata for filtering/retrieval
        """
        self.embedding_model = embedding_model
        self.config = config
        self.metadata_fields = metadata_fields or [
            "title",
            "paper_id",
            "page",
            "chunk_id",
        ]
        self.initialization_time = time.time()
        logger.info("Vectorstore initialized at: %s", self.initialization_time)

        # Track loaded papers to prevent duplicate loading
        self.loaded_papers = set()
        self.vector_store_class = FAISS
        logger.info("Using FAISS vector store")

        # Store for initialized documents
        self.documents: Dict[str, Document] = {}
        self.vector_store: Optional[VectorStore] = None
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}
        # Cache for document chunk embeddings to avoid recomputation
        self.embeddings: Dict[str, Any] = {}

    def add_paper(
        self,
        paper_id: str,
        pdf_url: str,
        paper_metadata: Dict[str, Any],
    ) -> None:
        """
        Add a paper to the document store.

        Args:
            paper_id: Unique identifier for the paper
            pdf_url: URL to the PDF
            paper_metadata: Metadata about the paper
        """
        # Skip if already loaded
        if paper_id in self.loaded_papers:
            logger.info("Paper %s already loaded, skipping", paper_id)
            return

        logger.info("Loading paper %s from %s", paper_id, pdf_url)

        # Store paper metadata
        self.paper_metadata[paper_id] = paper_metadata

        # Load the PDF and split into chunks according to Hydra config
        loader = PyPDFLoader(pdf_url)
        documents = loader.load()
        logger.info("Loaded %d pages from %s", len(documents), paper_id)

        # Create text splitter according to provided configuration
        if self.config is None:
            raise ValueError(
                "Configuration is required for text splitting in Vectorstore."
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Split documents and add metadata for each chunk
        chunks = splitter.split_documents(documents)
        logger.info("Split %s into %d chunks", paper_id, len(chunks))
        # Embed and cache chunk embeddings
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = self.embedding_model.embed_documents(chunk_texts)
        logger.info("Embedded %d chunks for paper %s", len(chunk_embeddings), paper_id)

        # Enhance document metadata
        for i, chunk in enumerate(chunks):
            # Add paper metadata to each chunk
            chunk.metadata.update(
                {
                    "paper_id": paper_id,
                    "title": paper_metadata.get("Title", "Unknown"),
                    "chunk_id": i,
                    # Keep existing page number if available
                    "page": chunk.metadata.get("page", 0),
                }
            )

            # Add any additional metadata fields
            for field in self.metadata_fields:
                if field in paper_metadata and field not in chunk.metadata:
                    chunk.metadata[field] = paper_metadata[field]

            # Store chunk
            doc_id = f"{paper_id}_{i}"
            self.documents[doc_id] = chunk
            # Cache embedding if available
            if chunk_embeddings[i] is not None:
                self.embeddings[doc_id] = chunk_embeddings[i]

        # Mark as loaded to prevent duplicate loading
        self.loaded_papers.add(paper_id)
        logger.info("Added %d chunks from paper %s", len(chunks), paper_id)

    def build_vector_store(self) -> None:
        """
        Build the vector store from all loaded documents.
        Should be called after all papers are added.
        """
        if not self.documents:
            logger.warning("No documents added to build vector store")
            return

        if self.vector_store is not None:
            logger.info("Vector store already built, skipping")
            return

        # Create vector store from documents
        documents_list = list(self.documents.values())
        self.vector_store = self.vector_store_class.from_documents(
            documents=documents_list, embedding=self.embedding_model
        )
        logger.info("Built vector store with %d documents", len(documents_list))
