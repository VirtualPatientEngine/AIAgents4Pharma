#!/usr/bin/env python3
"""
question_and_answer: Tool for performing Q&A on PDF documents using retrieval augmented generation.

This module provides functionality to extract text from PDF binary data, split it into
chunks, retrieve relevant segments via a vector store, and generate an answer to a
user-provided question using a language model chain.
"""

import io
import re
import logging
import os
import tempfile
from typing import Annotated, Any, Dict

import hydra
from langchain.docstore.document import Document
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
from PyPDF2 import PdfReader

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def extract_text_from_pdf_data(pdf_bytes: bytes) -> str:
    """
    Extract text content from PDF binary data.

    This function uses PyPDF2 to read the provided PDF bytes and concatenates the text
    extracted from each page.

    Args:
        pdf_bytes (bytes): The binary data of the PDF document.

    Returns:
        str: The complete text extracted from the PDF.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text


def generate_answer(
    question: str,
    pdf_source: Any,  # Can be bytes or URL
    text_embedding_model: Embeddings,
    llm_model: BaseChatModel,
) -> Dict[str, Any]:
    """
    Generate an answer for a question using retrieval augmented generation on PDF content.
    """
    # Load configuration using Hydra.
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/question_and_answer=default"]
        )
        cfg = cfg.tools.question_and_answer
        logger.info("Loaded Question and Answer tool configuration.")
    logger.info(f"Processing PDF with question: {question}")

    # Handle different source types (binary data or URL)
    if isinstance(pdf_source, bytes):
        logger.info("Processing PDF from binary data")
        try:
            # For binary data, use PyPDF2 to extract text
            text = extract_text_from_pdf_data(pdf_source)

            # Create Document objects from text chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
            chunks = text_splitter.split_text(text)
            documents = [Document(page_content=chunk) for chunk in chunks]
            logger.info(f"Split PDF text into {len(documents)} chunks.")
        except Exception as e:
            # If binary processing fails, try saving to temp file and using PyPDFLoader
            logger.warning(
                f"Error extracting text from binary: {str(e)}. Trying with temp file..."
            )
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_source)
                tmp_path = tmp.name

            try:
                # Use PyPDFLoader with the temp file
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                logger.info(
                    f"Loaded {len(documents)} pages with PyPDFLoader from temp file"
                )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    elif isinstance(pdf_source, str):
        # For URL, use PyPDFLoader
        logger.info(f"Processing PDF from URL: {pdf_source}")
        loader = PyPDFLoader(pdf_source)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages with PyPDFLoader from URL")

    else:
        raise ValueError(
            "pdf_source must be either bytes (PDF binary) or str (PDF URL)"
        )

    # Create vector store and perform similarity search
    vector_store = InMemoryVectorStore.from_documents(documents, text_embedding_model)
    search_results = vector_store.similarity_search(question, k=cfg.num_retrievals)
    logger.info(f"Retrieved {len(search_results)} relevant document chunks")

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

    This tool extracts PDF content from the state, processes it using a retrieval-based approach,
    and generates an answer to the user's question. It supports both arXiv downloaded PDFs and
    Zotero library PDFs, selecting the most relevant PDF when multiple are available.

    Args:
        question (str): The question to answer based on PDF content.
        tool_call_id (str): Unique identifier for the current tool call.
        state (dict): Current state dictionary containing PDF data and required models.
            Expected keys:
            - "pdf_data": Dictionary containing PDF data (either arXiv or Zotero format)
            - "text_embedding_model": Model for generating embeddings
            - "llm_model": Language model for generating answers

    Returns:
        Dict[str, Any]: A dictionary wrapped in a Command that updates the conversation
            with either the answer or an error message.

    Raises:
        ValueError: If required components are missing or if PDF processing fails.
    """
    logger.info("Starting PDF Question and Answer tool using PDF data from state.")

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

    # Get PDF data from state
    pdf_data = state.get("pdf_data", {})
    if not pdf_data:
        error_msg = "No pdf_data found in state."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check if pdf_data has the arxiv_downloader format
    if "pdf_object" in pdf_data or "pdf_url" in pdf_data:
        # This is the arxiv_downloader format
        logger.info("Detected arxiv_downloader PDF format")
        pdf_bytes = pdf_data.get("pdf_object")
        pdf_url = pdf_data.get("pdf_url", "")
        pdf_filename = "ArXiv paper {}".format(pdf_data.get("arxiv_id", "unknown"))
    else:
        # This is the Zotero format - navigate the nested structure
        paper_keys = list(pdf_data.keys())
        if not paper_keys:
            error_msg = "No papers with PDFs found in pdf_data."
            logger.error(error_msg)
            raise ValueError(error_msg)

        selected_paper_key = None
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
                    logger.info(
                        f"Selected paper {paper_num} based on numerical reference"
                    )

            # 2. If not found by number, check for title matches
            if not selected_paper_key:
                # Create a mapping of papers to their score for best match
                paper_scores = {key: 0 for key in paper_keys}

                for key in paper_keys:
                    for _, att_data in pdf_data[key].items():
                        if "filename" in att_data:
                            filename = att_data["filename"]

                            # Extract title component (after the last dash if present)
                            if " - " in filename:
                                title_part = filename.split(" - ")[-1].replace(
                                    ".pdf", ""
                                )
                            else:
                                title_part = filename.replace(".pdf", "")

                            # Score: Length of longest common substring between title and question
                            common_words = set(title_part.lower().split()) & set(
                                question.lower().split()
                            )
                            paper_scores[key] = len(common_words)

                # Select the paper with the highest score (if any match at all)
                if max(paper_scores.values()) > 0:
                    selected_paper_key = max(paper_scores.items(), key=lambda x: x[1])[
                        0
                    ]

                    # Log which paper was selected and why
                    for _, att_data in pdf_data[selected_paper_key].items():
                        if "filename" in att_data:
                            logger.info(
                                f"Selected paper '{att_data['filename']}' with score {paper_scores[selected_paper_key]}"
                            )
                            break

        # If no specific paper was found, use the first one
        if not selected_paper_key:
            selected_paper_key = paper_keys[0]
            logger.info(
                "Using first paper as no specific paper was mentioned in question"
            )

        paper_key = selected_paper_key
        attachments = pdf_data[paper_key]

        attachment_keys = list(attachments.keys())
        if not attachment_keys:
            error_msg = "No PDF attachments found for paper {}.".format(paper_key)
            logger.error(error_msg)
            raise ValueError(error_msg)

        attachment_key = attachment_keys[0]  # Use the first attachment
        pdf_attachment = attachments[attachment_key]

        # Get the binary data and optional URL
        pdf_bytes = pdf_attachment.get("data")
        pdf_url = pdf_attachment.get("url", "")
        pdf_filename = pdf_attachment.get("filename", "unknown.pdf")

    # Make sure we have either binary data or URL
    if not pdf_bytes and not pdf_url:
        error_msg = "Neither PDF binary data nor URL is available."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log information about the PDF
    logger.info("Retrieved PDF: {}".format(pdf_filename))
    if pdf_bytes:
        logger.info("PDF size: {} bytes".format(len(pdf_bytes)))
    if pdf_url:
        logger.info("PDF URL: {}".format(pdf_url))

    try:
        # Try to process the PDF (prioritize binary data if available)
        pdf_source = pdf_bytes if pdf_bytes else pdf_url
        result = generate_answer(question, pdf_source, text_embedding_model, llm_model)

        # Format the answer for return
        answer_text = result.get("output_text", "No answer generated.")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Answer based on PDF '{}':\n\n{}".format(
                            pdf_filename, answer_text
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    except Exception as e:
        error_msg = "Error processing PDF: {}".format(str(e))
        logger.error(error_msg)
        raise ValueError(error_msg)
