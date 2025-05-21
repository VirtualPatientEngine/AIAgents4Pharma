"""
PDF Question & Answer Tool
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
        logger.info("Loaded Question and Answer tool configuration.")
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
