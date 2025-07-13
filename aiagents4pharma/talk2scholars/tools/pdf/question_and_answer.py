"""
LangGraph PDF Retrieval-Augmented Generation (RAG) Tool

This tool answers user questions by retrieving and ranking relevant text chunks from PDFs
and invoking an LLM to generate a concise, source-attributed response. It supports
single or multiple PDF sources—such as Zotero libraries, arXiv papers, or direct uploads.

Workflow:
  1. (Optional) Load PDFs from diverse sources into a FAISS vector store of embeddings.
  2. Rerank candidate papers using NVIDIA NIM semantic re-ranker.
  3. Retrieve top-K diverse text chunks via Maximal Marginal Relevance (MMR).
  4. Build a context-rich prompt combining retrieved chunks and the user question.
  5. Invoke the LLM to craft a clear answer with source citations.
  6. Return the answer in a ToolMessage for LangGraph to dispatch.
"""

import logging
import os
import time
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .utils.generate_answer import load_hydra_config
from .utils.retrieve_chunks import retrieve_relevant_chunks
from .utils.tool_helper import QAToolHelper

# Helper for managing state, vectorstore, reranking, and formatting
helper = QAToolHelper()
# Load configuration and start logging
config = load_hydra_config()

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class QuestionAndAnswerInput(BaseModel):
    """
    Pydantic schema for the PDF Q&A tool inputs.

    Fields:
      question: User's free-text query to answer based on PDF content.
      tool_call_id: LangGraph-injected call identifier for tracking.
      state: Shared agent state dict containing:
        - article_data: metadata mapping of paper IDs to info (e.g., 'pdf_url', title).
        - text_embedding_model: embedding model instance for chunk indexing.
        - llm_model: chat/LLM instance for answer generation.
        - vector_store: optional pre-built Vectorstore for retrieval.
    """

    question: str = Field(
        description="User question for generating a PDF-based answer."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    LangGraph tool for Retrieval-Augmented Generation over PDFs.

    Given a user question, this tool applies the following pipeline:
      1. Validates that embedding and LLM models, plus article metadata, are in state.
      2. Initializes or reuses a FAISS-based Vectorstore for PDF embeddings.
      3. Loads one or more PDFs (from Zotero, arXiv, uploads) as text chunks into the store.
      4. Uses NVIDIA NIM semantic re-ranker to select top candidate papers.
      5. Retrieves the most relevant and diverse text chunks via Maximal Marginal Relevance.
      6. Constructs an LLM prompt combining contextual chunks and the query.
      7. Invokes the LLM to generate an answer, appending source attributions.
      8. Returns a LangGraph Command with a ToolMessage containing the answer.

    Args:
      question (str): The free-text question to answer.
      state (dict): Injected agent state; must include:
        - article_data: mapping paper IDs → metadata (pdf_url, title, etc.)
        - text_embedding_model: embedding model instance.
        - llm_model: chat/LLM instance.
      tool_call_id (str): Internal identifier for this tool invocation.

    Returns:
      Command[Any]: updates conversation state with a ToolMessage(answer).

    Raises:
      ValueError: when required models or metadata are missing in state.
      RuntimeError: when no relevant chunks can be retrieved for the query.
    """
    call_id = f"qa_call_{time.time()}"
    logger.info(
        "Starting PDF Question and Answer tool call %s for question: %s",
        call_id,
        question,
    )
    helper.start_call(config, call_id)

    # Extract models and article metadata
    text_emb, llm_model, article_data = helper.get_state_models_and_data(state)

    # Initialize or reuse vector store, then load candidate papers
    vs = helper.init_vector_store(text_emb)
    candidate_ids = list(article_data.keys())
    logger.info("%s: Candidate paper IDs for reranking: %s", call_id, candidate_ids)
    helper.load_candidate_papers(vs, article_data, candidate_ids)

    # Rerank papers and retrieve top chunks
    selected_ids = helper.run_reranker(vs, question, candidate_ids)
    relevant_chunks = retrieve_relevant_chunks(
        vs, query=question, paper_ids=selected_ids, top_k=config.top_k_chunks
    )
    if not relevant_chunks:
        msg = f"No relevant chunks found for question: '{question}'"
        logger.warning("%s: %s", call_id, msg)
        raise RuntimeError(msg)

    # Generate answer and format with sources
    response_text = helper.format_answer(
        question, relevant_chunks, llm_model, article_data
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
