#!/usr/bin/env python3
"""
Tool for performing Q&A on PDF documents using retrieval augmented generation.
This module provides functionality to load PDFs from URLs, split them into
chunks, retrieve relevant segments via a vector store, and generate an answer to a
user-provided question using a language model chain.
"""

import logging
import re
from typing import Annotated, Any, Dict

import hydra
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches, too-many-statements
# pylint: disable=broad-exception-caught


class QuestionAndAnswerInput(BaseModel):
    """
    Input schema for the PDF Question and Answer tool.

    Attributes:
        question (str): The question to ask regarding the PDF content.
        tool_call_id (str): Unique identifier for the tool call, injected automatically.
    """

    question: str = Field(description="The question to ask regarding the PDF content.")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


def generate_answer(
    question: str,
    pdf_url: str,
    text_embedding_model: Embeddings,
    llm_model: BaseChatModel,
) -> Dict[str, Any]:
    """
    Generate an answer for a question using retrieval augmented generation on PDF content.

    Args:
        question (str): The question to answer
        pdf_url (str): URL of the PDF to process
        text_embedding_model (Embeddings): Model for generating embeddings
        llm_model (BaseChatModel): Language model for generating answers

    Returns:
        Dict[str, Any]: Dictionary with the answer
    """
    # Load configuration using Hydra.
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/question_and_answer=default"]
        )
        cfg = cfg.tools.question_and_answer
        logger.info("Loaded Question and Answer tool configuration.")

    logger.info("Processing PDF with question: %s", question)
    logger.info("Processing PDF from URL: %s", pdf_url)

    # Use PyPDFLoader to load the PDF from URL
    loader = PyPDFLoader(pdf_url)
    documents = loader.load()
    logger.info("Loaded %d pages with PyPDFLoader from URL", len(documents))

    # For longer documents, split into smaller chunks
    if len(documents) > 1 or len(documents[0].page_content) > cfg.chunk_size:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        documents = text_splitter.split_documents(documents)
        logger.info("Split PDF text into %d chunks.", len(documents))

    # Create vector store and perform similarity search
    vector_store = InMemoryVectorStore.from_documents(documents, text_embedding_model)
    search_results = vector_store.similarity_search(question, k=cfg.num_retrievals)
    logger.info("Retrieved %d relevant document chunks", len(search_results))

    # Prepare context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in search_results])

    # Use the prompt template from Hydra config
    if hasattr(cfg, "prompt_template"):
        prompt = cfg.prompt_template.format(context=context, question=question)
    else:
        # Simple default format if prompt_template is not in config
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # Get the answer from the language model
    response = llm_model.invoke(prompt)

    # Return the response in the expected format
    return {"output_text": response.content}


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer_tool(
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command[Any]:
    """
    Answer a question using PDF content stored in the state via retrieval augmented generation.

    This tool retrieves PDF URLs from the state, loads and processes them using a
    retrieval-based approach, and generates an answer to the user's question.

    Args:
        question (str): The question to answer based on PDF content.
        tool_call_id (str): Unique identifier for the current tool call.
        state (dict): Current state dictionary containing article data and required models.
            Expected keys:
            - "article_data": Dictionary containing article metadata including PDF URLs
            - "text_embedding_model": Model for generating embeddings
            - "llm_model": Language model for generating answers

    Returns:
        Dict[str, Any]: A dictionary wrapped in a Command that updates the conversation
            with either the answer or an error message.

    Raises:
        ValueError: If required components are missing or if PDF processing fails.
    """
    logger.info("Starting PDF Question and Answer tool")

    # Get required models from state
    text_embedding_model = state.get("text_embedding_model")
    if not text_embedding_model:
        error_msg = "No text embedding model found in state."
        logger.error(error_msg)
        raise ValueError(error_msg)

    llm_model = state.get("llm_model")
    if not llm_model:
        error_msg = "No LLM model found in state."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Get article data from state
    article_data = state.get("article_data", {})
    if not article_data:
        error_msg = "No article_data found in state."
        logger.error(error_msg)
        raise ValueError(error_msg)

    paper_keys = list(article_data.keys())
    if not paper_keys:
        error_msg = "No papers found in article_data."
        logger.error(error_msg)
        raise ValueError(error_msg)

    selected_paper_key = None

    # If multiple papers are available, try to select the most relevant one
    if len(paper_keys) > 1:
        logger.info("Multiple papers found, trying to select the most relevant one")

        # 1. First try to match by paper number/position
        number_match = re.search(
            r"(?:^|\s)(?:(\d+)(?:st|nd|rd|th)?\s*(?:paper|pdf)|(?:paper|pdf)\s*(\d+))",
            question.lower(),
        )

        if number_match:
            # Get the number from whichever group matched (group 1 or group 2)
            paper_num_str = (
                number_match.group(1)
                if number_match.group(1)
                else number_match.group(2)
            )
            paper_num = int(paper_num_str)
            if 1 <= paper_num <= len(paper_keys):
                # Use 1-based indexing for user-friendly reference
                selected_paper_key = paper_keys[paper_num - 1]
                logger.info("Selected paper %s based on numerical reference", paper_num)

        # 2. If not found by number, check for title matches
        if not selected_paper_key:
            # Create a mapping of papers to their score for best match
            paper_scores = {key: 0 for key in paper_keys}

            for key in paper_keys:
                paper = article_data[key]
                title = paper.get("Title", "").lower()

                # Score: count of words from title appearing in question
                title_words = set(title.split())
                question_words = set(question.lower().split())
                common_words = title_words & question_words
                paper_scores[key] = len(common_words)

            # Select the paper with the highest score (if any match at all)
            if max(paper_scores.values()) > 0:
                selected_paper_key = max(paper_scores.items(), key=lambda x: x[1])[0]
                logger.info(
                    "Selected paper '%s' with score %s",
                    article_data[selected_paper_key].get("Title", "Unknown"),
                    paper_scores[selected_paper_key],
                )

    # If no specific paper was found, use the first one
    if not selected_paper_key:
        selected_paper_key = paper_keys[0]
        logger.info("Using first paper as no specific paper was mentioned in question")

    # Get the selected paper
    paper = article_data[selected_paper_key]

    # Get PDF URL from the paper
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        error_msg = (
            f"No PDF URL found for selected paper: {paper.get('Title', 'Unknown')}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Get the title or filename for reference
    paper_title = paper.get("Title", "Unknown")
    pdf_filename = paper.get("filename", paper_title)

    logger.info("Selected PDF: %s", pdf_filename)
    logger.info("PDF URL: %s", pdf_url)

    try:
        # Generate answer using the PDF URL
        result = generate_answer(question, pdf_url, text_embedding_model, llm_model)

        # Format the answer for return
        answer_text = result.get("output_text", "No answer generated.")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Answer based on PDF '{pdf_filename}':\n\n{answer_text}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
