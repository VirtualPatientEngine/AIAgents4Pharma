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
    candidate_ids = paper_ids or list(article_data.keys())
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
            ]
        }
    )
