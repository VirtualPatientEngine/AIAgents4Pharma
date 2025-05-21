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
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


# Shared pre-built Vectorstore for RAG (set externally, e.g., by Streamlit startup)
prebuilt_vector_store: Optional[Vectorstore] = None

def _get_state_models_and_data(state: dict, call_id: str):
    """Retrieve embedding model, LLM, and article data from agent state, or raise."""
    text_emb = state.get("text_embedding_model")
    if not text_emb:
        msg = "No text embedding model found in state."
        logger.error("%s: %s", call_id, msg)
        raise ValueError(msg)
    llm = state.get("llm_model")
    if not llm:
        msg = "No LLM model found in state."
        logger.error("%s: %s", call_id, msg)
        raise ValueError(msg)
    articles = state.get("article_data", {})
    if not articles:
        msg = "No article_data found in state."
        logger.error("%s: %s", call_id, msg)
        raise ValueError(msg)
    return text_emb, llm, articles

def _init_vector_store(emb_model, config):
    """Return shared prebuilt vector store or initialize a new one."""
    if prebuilt_vector_store is not None:
        logger.info("Using shared pre-built vector store from memory")
        return prebuilt_vector_store
    vs = Vectorstore(embedding_model=emb_model, config=config)
    logger.info("Initialized new vector store with provided configuration")
    return vs

def _load_candidate_papers(vs: Vectorstore, articles: dict, candidates: List[str], call_id: str):
    """Ensure each candidate paper is loaded into the vector store."""
    for pid in candidates:
        if pid not in vs.loaded_papers:
            pdf_url = articles.get(pid, {}).get("pdf_url")
            if not pdf_url:
                continue
            try:
                vs.add_paper(pid, pdf_url, articles[pid])
            except (IOError, ValueError) as exc:
                logger.warning("%s: Error loading paper %s: %s", call_id, pid, exc)

def _run_reranker(
    vs: Vectorstore,
    query: str,
    config: Any,
    candidates: List[str],
    call_id: str,
) -> List[str]:
    """Rank papers by relevance and return ordered paper IDs, falling back on candidates."""
    try:
        ranked = rank_papers_by_query(
            vs, query, config, top_k=config.top_k_papers
        )
        logger.info("%s: Papers after NVIDIA reranking: %s", call_id, ranked)
        return [pid for pid in ranked if pid in candidates]
    except (ValueError, RuntimeError) as exc:
        logger.error("%s: NVIDIA reranker failed: %s", call_id, exc)
        logger.info(
            "%s: Falling back to all %d candidate papers", call_id, len(candidates)
        )
        return candidates

def _format_answer(
    question: str,
    chunks: List[Any],
    llm: Any,
    config: Any,
    articles: dict,
) -> str:
    """Generate answer via LLM and format with source attribution."""
    result = generate_answer(question, chunks, llm, config)
    answer = result.get("output_text", "No answer generated.")
    titles = {}
    for pid in result.get("papers_used", []):
        if pid in articles:
            titles[pid] = articles[pid].get("Title", "Unknown paper")
    if titles:
        srcs = "\n\nSources:\n" + "\n".join(f"- {t}" for t in titles.values())
    else:
        srcs = ""
    logger.info(
        "Generated answer using %d chunks from %d papers",
        len(chunks),
        len(titles),
    )
    return f"{answer}{srcs}"


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    paper_ids: Optional[List[str]] = None,
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
            If not provided, all papers are considered and ranked by NVIDIA reranker.

    Returns:
        Command[Any]: A LangGraph Command that updates the conversation state:
            - 'messages': a single ToolMessage containing the generated answer text.

    Raises:
        ValueError: If required models or 'article_data' are missing from state.
        RuntimeError: If no relevant document chunks can be retrieved.
    """
    # Load config and initialize call ID
    config = load_hydra_config()
    call_id = f"qa_call_{time.time()}"
    logger.info("Starting PDF Question and Answer tool call %s for question: %s", call_id, question)

    # Retrieve models and metadata
    text_emb, llm_model, article_data = _get_state_models_and_data(state, call_id)

    # Initialize or reuse vector store and load papers
    vs = _init_vector_store(text_emb, config)
    candidate_ids = paper_ids or list(article_data.keys())
    logger.info("%s: Candidate paper IDs for reranking: %s", call_id, candidate_ids)
    _load_candidate_papers(vs, article_data, candidate_ids, call_id)

    # Rank papers and retrieve top chunks
    selected_ids = _run_reranker(
        vs, question, config, candidate_ids, call_id
    )
    relevant_chunks = retrieve_relevant_chunks(
        vs,
        query=question,
        paper_ids=selected_ids,
        top_k=config.top_k_chunks,
    )
    if not relevant_chunks:
        msg = f"No relevant chunks found for question: '{question}'"
        logger.warning("%s: %s", call_id, msg)
        raise RuntimeError(msg)

    # Generate and format the answer with sources
    response_text = _format_answer(
        question,
        relevant_chunks,
        llm_model,
        config,
        article_data,
    )
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
