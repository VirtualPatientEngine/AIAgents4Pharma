#!/usr/bin/env python3
"""
arxiv_paper_fetch: Tool for Fetching arXiv Papers and Downloading PDFs

This module connects to the arXiv API to retrieve metadata for a research paper and
download its corresponding PDF. It constructs an API query using a provided arXiv ID,
parses the XML response to locate the PDF link, and then downloads the PDF content.
The tool returns the PDF data along with metadata confirming the download operation.
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


class DownloadArxivPaperInput(BaseModel):
    """
    Input schema for the arXiv paper download tool.

    Attributes:
        arxiv_id (str): The unique identifier of the paper on arXiv.
        tool_call_id (str): A unique identifier automatically injected to track this tool call.
    """
    arxiv_id: str = Field(
        description="The arXiv paper ID used to retrieve the paper details and PDF."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    model_config = {"arbitrary_types_allowed": True}


# Load Hydra configuration for the download_arxiv_paper tool using a relative path.
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(
        config_name="config",
        overrides=["tools/download_arxiv_paper=default"]
    )
    cfg = cfg.tools.download_arxiv_paper


@tool(args_schema=DownloadArxivPaperInput, parse_docstring=True)
def download_arxiv_paper(
    arxiv_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Download an arXiv paper's PDF using its unique arXiv ID.

    This function performs the following operations:
      1. Connects to the arXiv API to fetch paper metadata based on the provided arXiv ID.
         It constructs an API URL, sends an HTTP GET request, and parses the XML response.
      2. Searches the parsed XML for a link labeled "pdf" to obtain the URL for the PDF.
      3. Downloads the PDF by streaming the content from the retrieved URL.
      4. Returns a Command object containing the binary PDF data, the PDF URL, and the arXiv ID,
         along with a confirmation message.

    Args:
        arxiv_id (str): The unique identifier for the paper on arXiv.
        tool_call_id (Annotated[str, InjectedToolCallId]):
        A unique tool call identifier for tracking purposes.

    Returns:
        Command[Any]: A command object with an update dictionary containing:
            - "pdf_data": A dictionary with keys "pdf_object" (the binary PDF content),
              "pdf_url" (the URL from which the PDF was downloaded),
               and "arxiv_id" (the provided paper ID).
            - "messages": A list containing a ToolMessage that confirms successful PDF download.

    Raises:
        RuntimeError: If the PDF link cannot be found in the fetched paper metadata.
    """
    api = cfg.api_url
    timeout = cfg.request_timeout
    logger.info("Starting download from arXiv with paper ID: %s", arxiv_id)

    # Construct the API URL using the paper ID and fetch metadata.
    api_url = f"{api}?search_query=id:{arxiv_id}&start=0&max_results=1"
    logger.info("Fetching metadata from: %s", api_url)
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()

    # Parse the XML response to locate the PDF link.
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

    # Read and assemble the PDF data from binary chunks.
    pdf_object = b"".join(
        chunk for chunk in pdf_response.iter_content(chunk_size=1024) if chunk
    )

    content = f"Successfully downloaded PDF for arXiv ID {arxiv_id} "

    # Return the Command with the PDF data and confirmation message.
    return Command(
        update={
            "pdf_data": {"pdf_object": pdf_object, "pdf_url": pdf_url, "arxiv_id": arxiv_id},
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
