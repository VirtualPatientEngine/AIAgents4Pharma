#!/usr/bin/env python3
"""
Tool for downloading arXiv paper metadata and retrieving the PDF URL.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Annotated, Any

import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadArxivPaperInput(BaseModel):
    """Input schema for the arXiv paper download tool."""

    arxiv_id: str = Field(
        description="The arXiv paper ID used to retrieve the paper details and PDF URL."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


def fetch_arxiv_metadata(
    api_url: str, arxiv_id: str, request_timeout: int
) -> ET.Element:
    """Fetch and parse metadata from the arXiv API."""
    query_url = f"{api_url}?search_query=id:{arxiv_id}&start=0&max_results=1"
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return ET.fromstring(response.text)


def extract_metadata(entry: ET.Element, ns: dict, arxiv_id: str) -> dict:
    """Extract metadata from the XML entry."""
    title_elem = entry.find("atom:title", ns)
    title = title_elem.text.strip() if title_elem is not None else "N/A"

    authors = [
        author_elem.find("atom:name", ns).text.strip()
        for author_elem in entry.findall("atom:author", ns)
        if author_elem.find("atom:name", ns) is not None
    ]

    summary_elem = entry.find("atom:summary", ns)
    abstract = summary_elem.text.strip() if summary_elem is not None else "N/A"

    published_elem = entry.find("atom:published", ns)
    pub_date = published_elem.text.strip() if published_elem is not None else "N/A"

    pdf_url = next(
        (
            link.attrib.get("href")
            for link in entry.findall("atom:link", ns)
            if link.attrib.get("title") == "pdf"
        ),
        None,
    )
    if not pdf_url:
        raise RuntimeError(f"Could not find PDF URL for arXiv ID {arxiv_id}")

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{arxiv_id}.pdf",
        "source": "arxiv",
        "arxiv_id": arxiv_id,
    }


@tool(args_schema=DownloadArxivPaperInput, parse_docstring=True)
def download_arxiv_paper(
    arxiv_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URL for an arXiv paper using its unique arXiv ID.
    """
    logger.info("Fetching metadata from arXiv for paper ID: %s", arxiv_id)

    # Load configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_arxiv_paper=default"]
        )
        api_url = cfg.tools.download_arxiv_paper.api_url
        request_timeout = cfg.tools.download_arxiv_paper.request_timeout

    # Fetch and parse metadata
    root = fetch_arxiv_metadata(api_url, arxiv_id, request_timeout)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"No entry found for arXiv ID {arxiv_id}")

    # Extract metadata
    metadata = extract_metadata(entry, ns, arxiv_id)

    # Create article_data entry with the paper ID as the key
    article_data = {arxiv_id: metadata}

    content = f"Successfully retrieved metadata and PDF URL for arXiv ID {arxiv_id}"

    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
