"""
Enhanced tool for performing Q&A on PDF documents using retrieval augmented generation.
This module provides functionality to load PDFs from URLs, split them into
chunks, retrieve relevant segments via semantic search, and generate answers
to user-provided questions using a language model chain.
"""

import logging
import os
import time
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from numpy import ndarray
from pydantic import BaseModel, Field

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class QuestionAndAnswerInput(BaseModel):
    """
    Input schema for the PDF Question and Answer tool.

    Attributes:
        question (str): The question to ask regarding the PDF content.
        paper_ids (Optional[List[str]]): Optional list of specific paper IDs to query.
        tool_call_id (str): Unique identifier for the tool call, injected automatically.
    """

    question: str = Field(description="The question to ask regarding the PDF content.")
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific paper IDs to query. If not provided, relevant papers will be selected automatically.",
    )
    use_all_papers: bool = Field(
        default=False,
        description="Whether to use all available papers for answering the question.",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


class DocumentStore:
    """
    A class for managing document embeddings and retrieval.
    Provides unified access to documents across multiple papers.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        metadata_fields: List[str] = None,
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
        logger.info(f"DocumentStore initialized at: {self.initialization_time}")

        # Track loaded papers to prevent duplicate loading
        self.loaded_papers = set()

        # Try to import FAISS, fall back to InMemoryVectorStore
        try:
            from langchain_community.vectorstores import FAISS

            self.vector_store_class = FAISS
            logger.info("Using FAISS vector store")
        except ImportError:
            from langchain_core.vectorstores import InMemoryVectorStore

            self.vector_store_class = InMemoryVectorStore
            logger.info("Using InMemoryVectorStore (FAISS not available)")

        # Store for initialized documents
        self.documents: Dict[str, Document] = {}
        self.vector_store: Optional[VectorStore] = None
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}

    def add_paper(
        self,
        paper_id: str,
        pdf_url: str,
        paper_metadata: Dict[str, Any],
        splitter: Optional[RecursiveCharacterTextSplitter] = None,
    ) -> None:
        """
        Add a paper to the document store.

        Args:
            paper_id: Unique identifier for the paper
            pdf_url: URL to the PDF
            paper_metadata: Metadata about the paper
            splitter: Text splitter to use (optional)
        """
        # Skip if already loaded
        if paper_id in self.loaded_papers:
            logger.info(f"Paper {paper_id} already loaded, skipping")
            return

        logger.info(f"Loading paper {paper_id} from {pdf_url}")

        # Store paper metadata
        self.paper_metadata[paper_id] = paper_metadata

        try:
            # Use PyPDFLoader to load the PDF
            loader = PyPDFLoader(pdf_url)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {paper_id}")

            # Use default splitter if none provided
            if splitter is None:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )

            # Split documents and add metadata
            chunks = splitter.split_documents(documents)
            logger.info(f"Split {paper_id} into {len(chunks)} chunks")

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

                # Store document
                doc_id = f"{paper_id}_{i}"
                self.documents[doc_id] = chunk

            # Mark as loaded to prevent duplicate loading
            self.loaded_papers.add(paper_id)
            logger.info(f"Added {len(chunks)} chunks from paper {paper_id}")

        except Exception as e:
            logger.error(f"Error loading paper {paper_id}: {str(e)}")
            raise

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
        logger.info(f"Built vector store with {len(documents_list)} documents")

    def rank_papers_by_query(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Rank papers by relevance to the query using semantic similarity.

        Args:
            query: The query string
            top_k: Number of top papers to return

        Returns:
            List of tuples (paper_id, score) sorted by relevance
        """
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Average embeddings by paper
        paper_scores = {}

        if self.vector_store:
            # Get embeddings for each document
            for doc_id, doc in self.documents.items():
                paper_id = doc.metadata["paper_id"]

                # Try to get embedding from vector store or recompute
                try:
                    # Different vector stores have different ways to access embeddings
                    if hasattr(self.vector_store, "index_to_docstore_id"):
                        # FAISS pattern
                        for (
                            i,
                            stored_id,
                        ) in self.vector_store.index_to_docstore_id.items():
                            if stored_id == doc_id:
                                doc_embedding = self.vector_store.index.reconstruct(i)
                                break
                    else:
                        # Fall back to recomputing
                        raise AttributeError(
                            "Vector store doesn't support embedding retrieval"
                        )
                except (AttributeError, KeyError):
                    # Recompute if needed
                    doc_embedding = self.embedding_model.embed_documents(
                        [doc.page_content]
                    )[0]

                    # Compute similarity score
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)

                    # Update paper score
                    if paper_id not in paper_scores:
                        paper_scores[paper_id] = {"scores": [], "avg": 0}

                    paper_scores[paper_id]["scores"].append(similarity)

        # Compute average score for each paper
        for paper_id, data in paper_scores.items():
            data["avg"] = np.mean(data["scores"])

        # Sort papers by average score
        sorted_papers = sorted(
            [(paper_id, data["avg"]) for paper_id, data in paper_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_papers[:top_k]

    def retrieve_relevant_chunks(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        top_k: int = 10,
        use_mmr: bool = True,
        mmr_diversity: float = 0.3,
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query, optionally filtering by paper_ids.

        Args:
            query: Query string
            paper_ids: Optional list of paper IDs to filter by
            top_k: Number of chunks to retrieve
            use_mmr: Whether to use maximal marginal relevance to ensure diversity
            mmr_diversity: Diversity parameter for MMR (higher = more diverse)

        Returns:
            List of document chunks
        """
        if not self.vector_store:
            logger.warning("Vector store not built, building now...")
            self.build_vector_store()

        if not self.vector_store:
            logger.error("Failed to build vector store")
            return []

        # Filter by paper_ids if provided
        metadata_filter = None
        if paper_ids:
            metadata_filter = {"paper_id": {"$in": paper_ids}}
            logger.info(f"Filtering retrieval to papers: {paper_ids}")

        # Retrieve using MMR or standard similarity
        if use_mmr:
            try:
                # Get embeddings
                query_embedding = np.array(self.embedding_model.embed_query(query))

                # Get document embeddings
                doc_embeddings = []
                docs = []

                for doc in self.documents.values():
                    # Apply filter if needed
                    if metadata_filter and doc.metadata["paper_id"] not in paper_ids:
                        continue

                    # Get document embedding
                    doc_embedding = np.array(
                        self.embedding_model.embed_documents([doc.page_content])[0]
                    )
                    doc_embeddings.append(doc_embedding)
                    docs.append(doc)

                # Apply MMR
                mmr_indices = maximal_marginal_relevance(
                    query_embedding,
                    np.array(doc_embeddings),
                    k=top_k,
                    lambda_mult=mmr_diversity,
                )

                results = [docs[i] for i in mmr_indices]
                logger.info(f"Retrieved {len(results)} chunks using MMR")
                return results

            except Exception as e:
                logger.warning(
                    f"MMR retrieval failed: {e}, falling back to standard similarity"
                )
                use_mmr = False

        if not use_mmr:
            # Use standard similarity search
            results = self.vector_store.similarity_search(
                query, k=top_k, filter=metadata_filter
            )
            logger.info(f"Retrieved {len(results)} chunks using similarity search")
            return results

    @staticmethod
    def _cosine_similarity(
        a: Union[List[float], ndarray], b: Union[List[float], ndarray]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(a, b) / (norm_a * norm_b)


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
    # Load configuration using Hydra if not provided
    if config is None:
        try:
            with hydra.initialize(version_base=None, config_path="../../configs"):
                cfg = hydra.compose(
                    config_name="config",
                    overrides=["tools/question_and_answer=default"],
                )
                config = cfg.tools.question_and_answer
                logger.info("Loaded Question and Answer tool configuration.")
        except Exception as e:
            logger.warning(f"Failed to load Hydra config: {e}. Using default values.")
            config = {}

    # Prepare context from retrieved documents with source attribution
    formatted_chunks = []

    for i, doc in enumerate(retrieved_chunks):
        # Extract metadata for attribution
        paper_id = doc.metadata.get("paper_id", "unknown")
        title = doc.metadata.get("title", "Unknown")
        page = doc.metadata.get("page", "unknown")

        # Format chunk with source information
        chunk_text = f"[Document {i+1}] From: '{title}' (ID: {paper_id}, Page: {page})\n{doc.page_content}"
        formatted_chunks.append(chunk_text)

    # Join all chunks
    context = "\n\n".join(formatted_chunks)

    # Get unique paper sources
    paper_sources = {doc.metadata["paper_id"] for doc in retrieved_chunks}

    # Use the prompt template from config or a comprehensive default
    if hasattr(config, "prompt_template") and config.prompt_template:
        prompt = config.prompt_template.format(context=context, question=question)
    else:
        # Enhanced default prompt
        prompt = f"""Answer the following question based on the provided context. 
If the context doesn't contain enough information to give a complete answer, say so.
Cite specific sources in your answer using the document numbers (e.g., [Document 1]).

Context:
{context}

Question: {question}

Your answer should be comprehensive, accurate, and well-structured.
"""

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
def question_and_answer_tool(
    question: str,
    paper_ids: Optional[List[str]] = None,
    use_all_papers: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = {},
) -> Union[Command[Any], Dict[str, Any]]:
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
            - "document_store": Optional DocumentStore instance

    Returns:
        Dict[str, Any]: A dictionary wrapped in a Command that updates the conversation
            with either the answer or an error message.

    Raises:
        ValueError: If required components are missing or if PDF processing fails.
    """
    # Create a unique identifier for this call to track potential infinite loops
    call_id = f"qa_call_{time.time()}"
    logger.info(
        f"Starting PDF Question and Answer tool call {call_id} for question: {question}"
    )

    # Get required models from state
    text_embedding_model = state.get("text_embedding_model")
    if not text_embedding_model:
        error_msg = "No text embedding model found in state."
        logger.error(f"{call_id}: {error_msg}")
        raise ValueError(error_msg)

    llm_model = state.get("llm_model")
    if not llm_model:
        error_msg = "No LLM model found in state."
        logger.error(f"{call_id}: {error_msg}")
        raise ValueError(error_msg)

    # Get article data from state
    article_data = state.get("article_data", {})
    if not article_data:
        error_msg = "No article_data found in state."
        logger.error(f"{call_id}: {error_msg}")
        raise ValueError(error_msg)

    # Get or create document store
    document_store = state.get("document_store")
    doc_store_created = False

    if not document_store:
        logger.info(f"{call_id}: Creating new document store")
        document_store = DocumentStore(embedding_model=text_embedding_model)
        doc_store_created = True
    else:
        logger.info(
            f"{call_id}: Using existing document store created at {getattr(document_store, 'initialization_time', 'unknown')}"
        )

    # Choose papers to use
    selected_paper_ids = []

    if paper_ids:
        # Use explicitly specified papers
        selected_paper_ids = [pid for pid in paper_ids if pid in article_data]
        logger.info(
            f"{call_id}: Using explicitly specified papers: {selected_paper_ids}"
        )

        if not selected_paper_ids:
            logger.warning(
                f"{call_id}: None of the provided paper_ids {paper_ids} were found"
            )

    elif use_all_papers:
        # Use all available papers
        selected_paper_ids = list(article_data.keys())
        logger.info(f"{call_id}: Using all {len(selected_paper_ids)} available papers")

    else:
        # Use semantic ranking to find relevant papers
        # First ensure papers are loaded
        for paper_id, paper in article_data.items():
            pdf_url = paper.get("pdf_url")
            if pdf_url and paper_id not in document_store.loaded_papers:
                try:
                    document_store.add_paper(paper_id, pdf_url, paper)
                except Exception as e:
                    logger.warning(f"{call_id}: Error loading paper {paper_id}: {e}")

        # Build vector store if needed
        if not document_store.vector_store:
            document_store.build_vector_store()

        # Now rank papers
        ranked_papers = document_store.rank_papers_by_query(question, top_k=3)
        selected_paper_ids = [paper_id for paper_id, _ in ranked_papers]
        logger.info(
            f"{call_id}: Selected papers based on semantic relevance: {selected_paper_ids}"
        )

    if not selected_paper_ids:
        # Fallback to all papers if selection failed
        selected_paper_ids = list(article_data.keys())
        logger.info(f"{call_id}: Falling back to all {len(selected_paper_ids)} papers")

    # Load selected papers if needed
    for paper_id in selected_paper_ids:
        if paper_id not in document_store.loaded_papers:
            pdf_url = article_data[paper_id].get("pdf_url")
            if pdf_url:
                try:
                    document_store.add_paper(paper_id, pdf_url, article_data[paper_id])
                except Exception as e:
                    logger.warning(f"{call_id}: Error loading paper {paper_id}: {e}")

    # Store the document store in state if it was created in this call
    if doc_store_created:
        # Use direct assignment to prevent recursive issues
        logger.info(f"{call_id}: Storing new document_store in state")
        state["document_store"] = document_store

    try:
        # Ensure vector store is built
        if not document_store.vector_store:
            document_store.build_vector_store()

        # Retrieve relevant chunks across selected papers
        relevant_chunks = document_store.retrieve_relevant_chunks(
            query=question, paper_ids=selected_paper_ids, top_k=10, use_mmr=True
        )

        if not relevant_chunks:
            error_msg = "No relevant chunks found in the papers."
            logger.warning(f"{call_id}: {error_msg}")

            if tool_call_id:
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
            f"{call_id}: Successfully generated answer using {len(relevant_chunks)} chunks from {len(paper_titles)} papers"
        )

        # Return as Command
        if tool_call_id:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=response_text,
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        else:
            # For direct calling
            return {"output": response_text, "metadata": result}

    except Exception as e:
        error_msg = f"Error processing PDFs: {str(e)}"
        logger.error(f"{call_id}: {error_msg}")

        if tool_call_id:
            raise RuntimeError(
                f"I encountered an error while processing your question: {error_msg}"
            )
        else:
            raise ValueError(error_msg) from e
