# File: aiagents4pharma/talk2scholars/tools/paper_download/download_pubmed_input.py
"""
This module defines the `download_pubmed_paper` tool, which leverages the
`PubmedPaperDownloader` class to fetch and download academic papers from pubmed
based on their unique pmID.
"""
from typing import Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from .pubmed_downloader import PubMedPaperDownloader

class DownloadPubMedPaperInput(BaseModel):
    """
    Input schema for the pubmed paper download tool.
    (Optional: if you decide to keep Pydantic validation in the future)
    """
    pmid: str = Field(
        description="The PubMed ID (PMID) used to retrieve the article."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool(args_schema=DownloadPubMedPaperInput)
def download_pubmed_paper(
    pmid: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Download a research article from PubMed using its PubMed ID (PMID).

    Args:
        pmid (str): The PubMed ID to retrieve the article.
        tool_call_id (str): The injected LangChain tool call ID.

    Returns:
        Command[Any]: A LangGraph command with PDF metadata and success message.
    """
    downloader = PubMedPaperDownloader()
    pdf_data = downloader.download_pdf(pmid)

    content = f"Successfully downloaded PDF for PMID {pmid}"

    return Command(
        update={
            "pdf_data": pdf_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
