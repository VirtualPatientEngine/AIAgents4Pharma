"""
LangGraph PDF Retrieval-Augmented Generation (RAG) Tool

This tool answers user questions using the traditional RAG pipeline:
1. Retrieve relevant chunks from ALL papers in the vector store
2. Rerank chunks using NVIDIA NIM reranker to find the most relevant ones
3. Generate answer using the top reranked chunks

Traditional RAG Pipeline Flow:
  Query → Retrieve chunks from ALL papers → Rerank chunks → Generate answer

This ensures the best possible chunks are selected across all available papers,
not just from pre-selected papers.
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

from .utils.answer_formatter import format_answer
from .utils.generate_answer import load_hydra_config
from .utils.paper_loader import load_all_papers
from .utils.rag_pipeline import retrieve_and_rerank_chunks
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
    """

    question: str = Field(description="User question for generating a PDF-based answer.")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    LangGraph tool for Retrieval-Augmented Generation over PDFs using traditional RAG pipeline.

    Traditional RAG Pipeline Implementation:
      1. Load ALL available PDFs into Milvus vector store (if not already loaded)
      2. Retrieve relevant chunks from ALL papers using vector similarity search
      3. Rerank retrieved chunks using NVIDIA NIM semantic reranker
      4. Generate answer using top reranked chunks with source attribution

    This approach ensures the best chunks are selected across all available papers,
    rather than pre-selecting papers and potentially missing relevant information.

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
        "Starting PDF Question and Answer tool (Traditional RAG Pipeline) - Call %s",
        call_id,
    )
    logger.info("%s: Question: '%s'", call_id, question)

    helper.start_call(config, call_id)

    # Extract models and article metadata
    text_emb, llm_model, article_data = helper.get_state_models_and_data(state)

    # Initialize or reuse Milvus vector store
    logger.info("%s: Initializing vector store", call_id)
    vs = helper.init_vector_store(text_emb)

    # Load ALL papers (traditional RAG approach)
    logger.info(
        "%s: Loading all %d papers into vector store (traditional RAG approach)",
        call_id,
        len(article_data),
    )
    load_all_papers(
        vector_store=vs,
        articles=article_data,
        call_id=call_id,
        config=config,
        has_gpu=helper.has_gpu,
    )

    # Traditional RAG Pipeline: Retrieve from ALL papers, then rerank
    logger.info(
        "%s: Starting traditional RAG pipeline: retrieve → rerank → generate",
        call_id,
    )

    # Retrieve and rerank chunks in one step
    reranked_chunks = retrieve_and_rerank_chunks(vs, question, config, call_id, helper.has_gpu)

    if not reranked_chunks:
        msg = f"No relevant chunks found for question: '{question}'"
        logger.warning("%s: %s", call_id, msg)

    # Generate answer using reranked chunks
    logger.info(
        "%s: Generating answer using %d reranked chunks",
        call_id,
        len(reranked_chunks),
    )
    response_text = format_answer(
        question,
        reranked_chunks,
        llm_model,
        article_data,
        config,
        call_id=call_id,
        has_gpu=helper.has_gpu,
    )

    logger.info(
        "%s: Successfully traditional completed RAG pipeline",
        call_id,
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
