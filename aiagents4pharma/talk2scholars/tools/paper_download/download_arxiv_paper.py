#!/usr/bin/env python3
"""
arxiv_paper_fetch: Tool for Fetching arXiv Papers and Downloading PDFs

This module connects to the arXiv API to get details about a research paper and download its PDF.
It uses network requests and XML parsing to find the PDF link for the paper.
"""
import logging
from typing import Annotated, Any
import xml.etree.ElementTree as ET
import requests
import hydra
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class downloadarxivpaperinput(BaseModel):
    """Input for fetching an arXiv paper.

    Attributes:
        arxiv_id (str): The unique ID of the paper on arXiv.
        tool_call_id (str): A unique identifier for this tool call.
    """

    arxiv_id: str = Field(
        description="The paper ID to fetch a paper."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    model_config = {"arbitrary_types_allowed": True}


# Use an absolute config path relative to this file's location.
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(
        config_name="config",
        overrides=["tools/download_arxiv_paper=default"]
    )
    cfg = cfg.tools.download_arxiv_paper


@tool(args_schema=downloadarxivpaperinput, parse_docstring=True)
def download_arxiv_paper(
    arxiv_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:

    """
    Download an arXiv paper and its PDF.

    This function does two things:
      1. It connects to the arXiv API to get the paper details and finds the PDF link.
      2. It downloads the PDF from the found link.

    Parameters:
        arxiv_id (str): The unique ID of the paper on arXiv.
        tool_call_id (str): A unique identifier for this call.

    Returns:
        Command[Any]: A command that contains:
            - The PDF data along with its URL and the paper's arXiv ID.
            - A message that confirms the PDF was downloaded successfully.

    Raises:
        RuntimeError: If it cannot find the PDF link in the paper details.
    """
    api = cfg.api_url
    timeout = cfg.request_timeout
    logger.info("Starting download from arXiv with paper ID: %s", arxiv_id)

    # Construct the API URL using the paper ID and fetch metadata.
    api_url = f"{api}?search_query=id:{arxiv_id}&start=0&max_results=1"
    logger.info("Fetching metadata from: %s", api_url)
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()

    # Parse the XML response.
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    pdf_url = next(
        (
            link.attrib.get("href")
            for entry in root.findall("atom:entry", ns)
            for link in entry.findall("atom:link", ns)
            if link.attrib.get("title") == "pdf"
        ),
        None,
    )

    if not pdf_url:
        raise RuntimeError(f"Failed to download PDF for arXiv ID {arxiv_id}.")

    logger.info("Downloading PDF from: %s", pdf_url)
    pdf_response = requests.get(pdf_url, stream=True, timeout=timeout)
    pdf_response.raise_for_status()

    # Read the PDF data as binary chunks and join them.
    pdf_object = b"".join(
        chunk for chunk in pdf_response.iter_content(chunk_size=1024) if chunk
    )

    content = f"Successfully downloaded PDF for arXiv ID {arxiv_id} "

    # Update the state by saving the PDF data along with its URL using the paper_id as the key.
    return Command(
        update={
            "pdf_data": {"pdf_object": pdf_object, "pdf_url": pdf_url,"arxiv_id": arxiv_id},
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
