#!/usr/bin/env python3
"""
qna: Tool for performing Q&A on PDF documents using retrieval augmented generation
"""
import io
import logging
from typing import Annotated, Dict, Any, List

from PyPDF2 import PdfReader
from pydantic import BaseModel, Field
import hydra
from omegaconf import OmegaConf

from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Annoy
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load configuration using Hydra.
with hydra.initialize(version_base=None, config_path="../../configs/tools/qna"):
    cfg = hydra.compose(config_name="default")
    logger.info(
        "Loaded QnA tool configuration:\n%s",
        OmegaConf.to_yaml(cfg)
    )

class QnaInput(BaseModel):
    """Input schema for the PDF QnA tool."""
    question: str = Field(description="The question to ask regarding the PDF content.")
    tool_call_id: Annotated[str, InjectedToolCallId]

def extract_text_from_pdf_data(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF data provided as bytes.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

def generate_answer(question: str, pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Generate an answer using retrieval augmented generation on the PDF data.
    """
    text = extract_text_from_pdf_data(pdf_bytes)
    logger.info("Extracted text from PDF.")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    documents: List[Document] = [Document(page_content=chunk) for chunk in chunks]
    logger.info("Split PDF text into %d chunks.", len(documents))

    embeddings = OpenAIEmbeddings(openai_api_key=cfg.openai_api_key)
    vector_store = Annoy.from_documents(documents, embeddings)
    search_results = vector_store.similarity_search(
        question,
        k=cfg.num_retrievals
    )
    logger.info("Retrieved %d relevant document chunks.", len(search_results))
    llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
    qa_chain = load_qa_chain(llm, chain_type=cfg.qa_chain_type)
    answer = qa_chain.invoke(
        input={"input_documents": search_results, "question": question}
    )
    return answer

@tool(args_schema=QnaInput)
def qna_tool(
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Dict[str, Any]:
    """
    Perform retrieval augmented generation on a PDF document stored in state (pdf_data)
    and answer a question based on its content.
    """
    logger.info("Starting PDF QnA tool using PDF data from state.")
    pdf_state = state.get("pdf_data")
    if not pdf_state:
        error_msg = "No pdf_data found in state."
        logger.error(error_msg)
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )
    pdf_bytes = pdf_state.get("pdf_object")
    if not pdf_bytes:
        error_msg = "PDF binary data is missing in the pdf_data from state."
        logger.error(error_msg)
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )
    answer = generate_answer(question, pdf_bytes)
    logger.info("Generated answer: %s", answer)
    return answer
