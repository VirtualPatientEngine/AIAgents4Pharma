#!/usr/bin/env python3
"""
arxiv_paper_fetch: Tool for fetching an arXiv paper by its arXiv ID and returning the PDF
as an object.

This tool interacts with the arXiv API to fetch metadata and download research papers as PDFs
based on their unique arXiv ID.
"""

import logging
from typing import Annotated, Any
import xml.etree.ElementTree as ET
import requests
import hydra
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FetchArxivPaperInput(BaseModel):
    """Input schema for the arXiv paper fetching tool."""

    paper_id: str = Field(
        description="The paper ID to fetch a paper."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    model_config = {"arbitrary_types_allowed": True}


# Use an absolute config path relative to this file's location.
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(
        config_name="config",
        overrides=["+tools/download_pdf_arxiv=default"]
    )
    cfg = cfg.tools.download_pdf_arxiv


@tool(args_schema=FetchArxivPaperInput, parse_docstring=True)
def fetch_arxiv_paper(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]= None,
) -> Command[Any]:
    if state is None:
        state = {}
    """
    Fetch an arXiv paper's metadata and download its PDF, returning the PDF as an object.

    Args:
        paper_id (str): The arXiv ID of the paper.
        tool_call_id (str): Unique tool call identifier.
        state (dict): The agent's state (can be used to store the downloaded PDF).

    Returns:
        Dict[str, Any]: A Command update containing the PDF object and a ToolMessage.
    """
    api = cfg.api_url
    timeout = cfg.request_timeout
    logger.info("Starting download from arXiv with paper ID: %s", paper_id)

    # Construct the API URL using the paper ID and fetch metadata.
    arxiv_id = state.get(paper_id, {}).get("arXiv_ID")
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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to download PDF for arXiv ID {paper_id}.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    logger.info("Downloading PDF from: %s", pdf_url)
    pdf_response = requests.get(pdf_url, stream=True, timeout=timeout)
    pdf_response.raise_for_status()

    # Read the PDF data as binary chunks and join them.
    pdf_data = b"".join(
        chunk for chunk in pdf_response.iter_content(chunk_size=1024) if chunk
    )

    content = f"Successfully downloaded PDF for arXiv ID {arxiv_id} and it is available in state."

    # Update the state by saving the PDF data along with its URL using the paper_id as the key.
    return Command(
        update={
            paper_id: {"pdf": pdf_data, "pdf_url": pdf_url},
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
