# File: aiagents4pharma/talk2scholars/tools/paper_download/download_arxiv_input.py
"""
This module defines the `download_arxiv_paper` tool, which leverages the
`ArxivPaperDownloader` class to fetch and download academic papers from arXiv
based on their unique arXiv ID.
"""
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

# Local import from the same package:
from .arxiv_downloader import ArxivPaperDownloader

class DownloadArxivPaperInput(BaseModel):
    """
    Input schema for the arXiv paper download tool.
    (Optional: if you decide to keep Pydantic validation in the future)
    """
    arxiv_id: str = Field(
        description="The arXiv paper ID used to retrieve the paper details and PDF."
        )
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool(args_schema=DownloadArxivPaperInput, parse_docstring=True)
def download_arxiv_paper(
    arxiv_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Download an arXiv paper's PDF using its unique arXiv ID.

    This function:
      1. Creates an `ArxivPaperDownloader` instance.
      2. Fetches metadata from arXiv using the provided `arxiv_id`.
      3. Downloads the PDF from the returned link.
      4. Returns a `Command` object containing the PDF data and a success message.

    Args:
        arxiv_id (str): The unique arXiv paper ID.
        tool_call_id (InjectedToolCallId): A unique identifier for tracking this tool call.

    Returns:
        Command[Any]: Contains metadata and messages about the success of the operation.
    """
    downloader = ArxivPaperDownloader()

    # If the downloader fails or the arxiv_id is invalid, this might raise an error
    pdf_data = downloader.download_pdf(arxiv_id)

    content = f"Successfully downloaded PDF for arXiv ID {arxiv_id}"

    return Command(
        update={
            "pdf_data": pdf_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
