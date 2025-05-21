"""
PDF Question & Answer Tool

This LangGraph tool answers user questions by leveraging a pre-built FAISS vector store
of embedded PDF document chunks. Given a question, it retrieves the most relevant text
segments from the loaded PDFs, invokes an LLM for answer generation, and returns the
response with source attribution.
"""

import logging
import os
import time
from typing import Annotated, Any, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .utils.generate_answer import generate_answer, load_hydra_config
from .utils.vector_store import Vectorstore
from .utils.retrieve_chunks import retrieve_relevant_chunks
from .utils.nvidia_nim_reranker import rank_papers_by_query

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class QuestionAndAnswerInput(BaseModel):
    """
    Input schema for the PDF Q&A tool.

    Attributes:
        question (str): Free-text question to answer based on PDF content.
        paper_ids (Optional[List[str]]): If provided, restricts retrieval to these paper IDs.
        use_all_papers (bool): If True, include all loaded papers without semantic ranking.
        tool_call_id (str): Internal ID injected by LangGraph for this tool call.
        state (dict): Shared agent state containing:
            - 'article_data': dict of paper metadata with 'pdf_url' keys
            - 'text_embedding_model': embedding model instance
            - 'llm_model': chat/LLM instance
            - 'vector_store': pre-built Vectorstore for retrieval
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


# Shared pre-built Vectorstore for RAG (set externally, e.g., by Streamlit startup)
prebuilt_vector_store: Optional[Vectorstore] = None


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    paper_ids: Optional[List[str]] = None,
    use_all_papers: bool = False,
) -> Command[Any]:
    """
    Generate an answer to a user question using Retrieval-Augmented Generation (RAG) over PDFs.

    This tool expects that a FAISS vector store of PDF document chunks has already been built
    and stored in shared state. It retrieves the most relevant chunks for the input question,
    invokes an LLM to craft a response, and returns the answer with source attribution.

    Args:
        question (str): The free-text question to answer.
        state (dict): Injected agent state mapping that must include:
            - 'article_data': mapping of paper IDs to metadata (including 'pdf_url')
            - 'text_embedding_model': the embedding model instance
            - 'llm_model': the chat/LLM instance
        tool_call_id (str): Internal identifier for this tool call.
        paper_ids (Optional[List[str]]): Specific paper IDs to restrict retrieval (default: None).
        use_all_papers (bool): If True, bypasses semantic ranking and includes all papers.

    Returns:
        Command[Any]: A LangGraph Command that updates the conversation state:
            - 'messages': a single ToolMessage containing the generated answer text.

    Raises:
        ValueError: If required models or 'article_data' are missing from state.
        RuntimeError: If no relevant document chunks can be retrieved.
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

    # Use shared pre-built Vectorstore if provided, else create a new one
    if prebuilt_vector_store is not None:
        vector_store = prebuilt_vector_store
        logger.info("Using shared pre-built vector store from the memory")
    else:
        vector_store = Vectorstore(
            embedding_model=text_embedding_model,
            config=config,
        )
        logger.info("Initialized new vector store with provided configuration")

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
        ranked_papers = rank_papers_by_query(
            vector_store,
            question,
            config,
            top_k=config.top_k_papers,
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
    relevant_chunks = retrieve_relevant_chunks(
        vector_store,
        query=question,
        paper_ids=selected_paper_ids,
        top_k=config.top_k_chunks,
    )

    if not relevant_chunks:
        error_msg = "No relevant chunks found in the papers."
        logger.warning("%s: %s", call_id, error_msg)
        raise RuntimeError(
            f"I couldn't find relevant information to answer your question: '{question}'. "
            "Please try rephrasing or asking a different question."
        )

    # Generate answer using retrieved chunks
    result = generate_answer(question, relevant_chunks, llm_model, config)

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
