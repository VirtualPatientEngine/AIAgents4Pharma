"""
Generate an answer for a question using retrieved chunks of documents.
"""

import logging
import os
from typing import Any, Dict, List

import hydra
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def _build_context_and_sources(
    retrieved_chunks: List[Document],
) -> tuple[str, set[str]]:
    """
    Build the combined context string and set of paper_ids from retrieved chunks.
    """
    papers = {}
    for doc in retrieved_chunks:
        pid = doc.metadata.get("paper_id", "unknown")
        papers.setdefault(pid, []).append(doc)
    formatted = []
    idx = 1
    for pid, chunks in papers.items():
        title = chunks[0].metadata.get("title", "Unknown")
        formatted.append(f"[Document {idx}] From: '{title}' (ID: {pid})")
        for chunk in chunks:
            page = chunk.metadata.get("page", "unknown")
            formatted.append(f"Page {page}: {chunk.page_content}")
        idx += 1
    context = "\n\n".join(formatted)
    sources: set[str] = set()
    for doc in retrieved_chunks:
        pid = doc.metadata.get("paper_id")
        if isinstance(pid, str):
            sources.add(pid)
    return context, sources


def load_hydra_config() -> Any:
    """
    Load the configuration using Hydra and return the configuration for the Q&A tool.
    """
    with hydra.initialize(version_base=None, config_path="../../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tools/question_and_answer=default"],
        )
        config = cfg.tools.question_and_answer
        logger.debug("Loaded Question and Answer tool configuration.")
        return config


def generate_answer(
    question: str,
    retrieved_chunks: List[Document],
    llm_model: BaseChatModel,
    config: Any,
) -> Dict[str, Any]:
    """
    Generate an answer for a question using retrieved chunks.

    Args:
        question (str): The question to answer
        retrieved_chunks (List[Document]): List of relevant document chunks
        llm_model (BaseChatModel): Language model for generating answers
        config (Any): Configuration for answer generation

    Returns:
        Dict[str, Any]: Dictionary with the answer and metadata
    """
    # Ensure the configuration is provided and has the prompt_template.
    if config is None:
        raise ValueError("Configuration for generate_answer is required.")
    if "prompt_template" not in config:
        raise ValueError("The prompt_template is missing from the configuration.")

    # Build context and sources, then invoke LLM
    context, paper_sources = _build_context_and_sources(retrieved_chunks)
    prompt = config["prompt_template"].format(context=context, question=question)
    response = llm_model.invoke(prompt)

    # Return the response with metadata
    return {
        "output_text": response.content,
        "sources": [doc.metadata for doc in retrieved_chunks],
        "num_sources": len(retrieved_chunks),
        "papers_used": list(paper_sources),
    }
