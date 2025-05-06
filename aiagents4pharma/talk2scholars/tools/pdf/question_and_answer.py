"""
Tool for performing Q&A on PDF documents using retrieval augmented generation.
This module provides functionality to load PDFs from URLs, split them into
chunks, retrieve relevant segments via semantic search, and generate answers
to user-provided questions using a language model chain.
"""

import logging
import os
import time
from typing import Annotated, Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))
# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements


def load_hydra_config() -> Any:
    """
    Load the configuration using Hydra and return the configuration for the Q&A tool.
    """
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tools/question_and_answer=default"],
        )
        config = cfg.tools.question_and_answer
        logger.info("Loaded Question and Answer tool configuration.")
        return config


class QuestionAndAnswerInput(BaseModel):
    """
    Input schema for the PDF Question and Answer tool.

    This schema defines the inputs required for querying academic or research-related
    PDFs to answer a specific question using a language model and document retrieval.

    Attributes:
        question (str): The question to ask regarding the PDF content.
        paper_ids (Optional[List[str]]): Optional list of specific paper IDs to query.
            If not provided, the system will determine relevant papers automatically.
        use_all_papers (bool): Whether to use all available papers for answering the question.
            If True, the system will include all loaded papers regardless of relevance filtering.
        tool_call_id (str): Unique identifier for the tool call, injected automatically.
        state (dict): Shared application state, injected automatically.
    """

    question: str = Field(description="The question to ask regarding the PDF content.")
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific paper IDs to query. "
        "If not provided, relevant papers will be selected automatically.",
    )
    use_all_papers: bool = Field(
        default=False,
        description="Whether to use all available papers for answering the question. "
        "Set to True to bypass relevance filtering and include all loaded papers.",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


class Vectorstore:
    """
    A class for managing document embeddings and retrieval.
    Provides unified access to documents across multiple papers.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        metadata_fields: Optional[List[str]] = None,
    ):
        """
        Initialize the document store.

        Args:
            embedding_model: The embedding model to use
            metadata_fields: Fields to include in document metadata for filtering/retrieval
        """
        self.embedding_model = embedding_model
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

        # Create text splitter according to Hydra config
        cfg = load_hydra_config()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Split documents and add metadata for each chunk
        chunks = splitter.split_documents(documents)
        logger.info("Split %s into %d chunks", paper_id, len(chunks))

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

    def rank_papers_by_query(
        self, query: str, top_k: int = 40
    ) -> List[Tuple[str, float]]:
        """
        Rank papers by relevance to the query using NVIDIA's off-the-shelf re-ranker.

        This function aggregates all chunks per paper, ranks them using the NVIDIA model,
        and returns the top-k papers.

        Args:
            query (str): The query string.
            top_k (int): Number of top papers to return.

        Returns:
            List of tuples (paper_id, dummy_score) sorted by relevance.
        """

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

        # Instantiate the NVIDIA re-ranker client
        config = load_hydra_config()
        reranker = NVIDIARerank(
            model=config.reranker.model,
            api_key=config.reranker.api_key,
        )

        # Get the ranked list of documents based on the query
        response = reranker.compress_documents(
            query=query, documents=aggregated_documents
        )

        ranked_papers = [doc.metadata["paper_id"] for doc in response[:top_k]]
        return ranked_papers

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
        logger.info(
            "Embedding query using model: %s", type(self.embedding_model).__name__
        )
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

        texts = [doc.page_content for doc in all_docs]

        # Step 3: Batch embed all documents
        logger.info("Starting batch embedding for %d chunks...", len(texts))
        all_embeddings = self.embedding_model.embed_documents(texts)
        logger.info("Completed embedding for %d chunks...", len(texts))

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


def generate_answer(
    question: str,
    retrieved_chunks: List[Document],
    llm_model: BaseChatModel,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate an answer for a question using retrieved chunks.

    Args:
        question (str): The question to answer
        retrieved_chunks (List[Document]): List of relevant document chunks
        llm_model (BaseChatModel): Language model for generating answers
        config (Optional[Any]): Configuration for answer generation

    Returns:
        Dict[str, Any]: Dictionary with the answer and metadata
    """
    # Load configuration using the global function.
    config = load_hydra_config()

    # Ensure the configuration is not None and has the prompt_template.
    if config is None:
        raise ValueError("Hydra config loading failed: config is None.")
    if "prompt_template" not in config:
        raise ValueError("The prompt_template is missing from the configuration.")

    # Prepare context from retrieved documents with source attribution.
    # Group chunks by paper_id
    papers = {}
    for doc in retrieved_chunks:
        paper_id = doc.metadata.get("paper_id", "unknown")
        if paper_id not in papers:
            papers[paper_id] = []
        papers[paper_id].append(doc)

    # Format chunks by paper
    formatted_chunks = []
    doc_index = 1
    for paper_id, chunks in papers.items():
        # Get the title from the first chunk (should be the same for all chunks)
        title = chunks[0].metadata.get("title", "Unknown")

        # Add a document header
        formatted_chunks.append(
            f"[Document {doc_index}] From: '{title}' (ID: {paper_id})"
        )

        # Add each chunk with its page information
        for chunk in chunks:
            page = chunk.metadata.get("page", "unknown")
            formatted_chunks.append(f"Page {page}: {chunk.page_content}")

        # Increment document index for the next paper
        doc_index += 1

    # Join all chunks
    context = "\n\n".join(formatted_chunks)

    # Get unique paper sources.
    paper_sources = {doc.metadata["paper_id"] for doc in retrieved_chunks}

    # Create prompt using the Hydra-provided prompt_template.
    prompt = config["prompt_template"].format(context=context, question=question)

    # Get the answer from the language model
    response = llm_model.invoke(prompt)

    # Return the response with metadata
    return {
        "output_text": response.content,
        "sources": [doc.metadata for doc in retrieved_chunks],
        "num_sources": len(retrieved_chunks),
        "papers_used": list(paper_sources),
    }


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    paper_ids: Optional[List[str]] = None,
    use_all_papers: bool = False,
) -> Command[Any]:
    """
    Answer a question using PDF content with advanced retrieval augmented generation.

    This tool retrieves PDF documents from URLs, processes them using semantic search,
    and generates an answer to the user's question based on the most relevant content.
    It can work with multiple papers simultaneously and provides source attribution.

    Args:
        question (str): The question to answer based on PDF content.
        paper_ids (Optional[List[str]]): Optional list of specific paper IDs to query.
        use_all_papers (bool): Whether to use all available papers.
        tool_call_id (str): Unique identifier for the current tool call.
        state (dict): Current state dictionary containing article data and required models.
            Expected keys:
            - "article_data": Dictionary containing article metadata including PDF URLs
            - "text_embedding_model": Model for generating embeddings
            - "llm_model": Language model for generating answers
            - "vector_store": Optional Vectorstore instance

    Returns:
        Dict[str, Any]: A dictionary wrapped in a Command that updates the conversation
            with either the answer or an error message.

    Raises:
        ValueError: If required components are missing or if PDF processing fails.
    """
    # Load configuration
    config = load_hydra_config()
    # Create a unique identifier for this call to track potential infinite loops
    call_id = f"qa_call_{time.time()}"
    logger.info(
        "Starting PDF Question and Answer tool call %s for question: %s",
        call_id,
        question,
    )

    # Get required models from state
    text_embedding_model = state.get("text_embedding_model")
    if not text_embedding_model:
        error_msg = "No text embedding model found in state."
        logger.error("%s: %s", call_id, error_msg)
        raise ValueError(error_msg)

    llm_model = state.get("llm_model")
    if not llm_model:
        error_msg = "No LLM model found in state."
        logger.error("%s: %s", call_id, error_msg)
        raise ValueError(error_msg)

    # Get article data from state
    article_data = state.get("article_data", {})
    if not article_data:
        error_msg = "No article_data found in state."
        logger.error("%s: %s", call_id, error_msg)
        raise ValueError(error_msg)

    # Always use a fresh in-memory document store for this Q&A call
    vector_store = Vectorstore(embedding_model=text_embedding_model)

    # Check if there are papers from different sources
    has_uploaded_papers = any(
        paper.get("source") == "upload"
        for paper in article_data.values()
        if isinstance(paper, dict)
    )

    has_zotero_papers = any(
        paper.get("source") == "zotero"
        for paper in article_data.values()
        if isinstance(paper, dict)
    )

    has_arxiv_papers = any(
        paper.get("source") == "arxiv"
        for paper in article_data.values()
        if isinstance(paper, dict)
    )

    # Choose papers to use
    selected_paper_ids = []

    if paper_ids:
        # Use explicitly specified papers
        selected_paper_ids = [pid for pid in paper_ids if pid in article_data]
        logger.info(
            "%s: Using explicitly specified papers: %s", call_id, selected_paper_ids
        )

        if not selected_paper_ids:
            logger.warning(
                "%s: None of the provided paper_ids %s were found", call_id, paper_ids
            )

    elif use_all_papers or has_uploaded_papers or has_zotero_papers or has_arxiv_papers:
        # Use all available papers if explicitly requested or if we have papers from any source
        selected_paper_ids = list(article_data.keys())
        logger.info(
            "%s: Using all %d available papers", call_id, len(selected_paper_ids)
        )

    else:
        # Use semantic ranking to find relevant papers
        # First ensure papers are loaded
        for paper_id, paper in article_data.items():
            pdf_url = paper.get("pdf_url")
            if pdf_url and paper_id not in vector_store.loaded_papers:
                try:
                    vector_store.add_paper(paper_id, pdf_url, paper)
                except (IOError, ValueError) as e:
                    logger.error("Error loading paper %s: %s", paper_id, e)
                    raise

        # Now rank papers
        ranked_papers = vector_store.rank_papers_by_query(
            question, top_k=config.top_k_papers
        )
        selected_paper_ids = [paper_id for paper_id, _ in ranked_papers]
        logger.info(
            "%s: Selected papers based on semantic relevance: %s",
            call_id,
            selected_paper_ids,
        )

    if not selected_paper_ids:
        # Fallback to all papers if selection failed
        selected_paper_ids = list(article_data.keys())
        logger.info(
            "%s: Falling back to all %d papers", call_id, len(selected_paper_ids)
        )

    # Load selected papers if needed
    for paper_id in selected_paper_ids:
        if paper_id not in vector_store.loaded_papers:
            pdf_url = article_data[paper_id].get("pdf_url")
            if pdf_url:
                try:
                    vector_store.add_paper(paper_id, pdf_url, article_data[paper_id])
                except (IOError, ValueError) as e:
                    logger.warning(
                        "%s: Error loading paper %s: %s", call_id, paper_id, e
                    )

    # Ensure vector store is built
    if not vector_store.vector_store:
        vector_store.build_vector_store()

    # Retrieve relevant chunks across selected papers
    relevant_chunks = vector_store.retrieve_relevant_chunks(
        query=question, paper_ids=selected_paper_ids, top_k=config.top_k_chunks
    )

    if not relevant_chunks:
        error_msg = "No relevant chunks found in the papers."
        logger.warning("%s: %s", call_id, error_msg)
        raise RuntimeError(
            f"I couldn't find relevant information to answer your question: '{question}'. "
            "Please try rephrasing or asking a different question."
        )

    # Generate answer using retrieved chunks
    result = generate_answer(question, relevant_chunks, llm_model)

    # Format answer with attribution
    answer_text = result.get("output_text", "No answer generated.")

    # Get paper titles for sources
    paper_titles = {}
    for paper_id in result.get("papers_used", []):
        if paper_id in article_data:
            paper_titles[paper_id] = article_data[paper_id].get(
                "Title", "Unknown paper"
            )

    # Format source information
    sources_text = ""
    if paper_titles:
        sources_text = "\n\nSources:\n" + "\n".join(
            [f"- {title}" for title in paper_titles.values()]
        )

    # Prepare the final response
    response_text = f"{answer_text}{sources_text}"
    logger.info(
        "%s: Successfully generated answer using %d chunks from %d papers",
        call_id,
        len(relevant_chunks),
        len(paper_titles),
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response_text,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
