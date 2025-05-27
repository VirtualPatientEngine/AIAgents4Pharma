#!/usr/bin/env python3
"""
Tool for downloading arXiv paper metadata and retrieving the PDF URL.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Annotated, Any, List

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

    arxiv_ids: List[str] = Field(
        description="List of arXiv paper IDs used to retrieve paper details and PDF URLs."
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
    title = (title_elem.text or "").strip() if title_elem is not None else "N/A"

    authors = []
    for author_elem in entry.findall("atom:author", ns):
        name_elem = author_elem.find("atom:name", ns)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    summary_elem = entry.find("atom:summary", ns)
    abstract = (summary_elem.text or "").strip() if summary_elem is not None else "N/A"

    published_elem = entry.find("atom:published", ns)
    pub_date = (
        (published_elem.text or "").strip() if published_elem is not None else "N/A"
    )

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


@tool(
    args_schema=DownloadArxivPaperInput,
    parse_docstring=True,
)
def download_arxiv_paper(
    arxiv_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URLs for one or more arXiv papers using their unique arXiv IDs.
    """
    logger.info("Fetching metadata from arXiv for paper IDs: %s", arxiv_ids)

    # Load configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_arxiv_paper=default"]
        )
        api_url = cfg.tools.download_arxiv_paper.api_url
        request_timeout = cfg.tools.download_arxiv_paper.request_timeout

    # Aggregate results
    article_data: dict[str, Any] = {}
    for aid in arxiv_ids:
        logger.info("Processing arXiv ID: %s", aid)
        # Fetch and parse metadata
        root = fetch_arxiv_metadata(api_url, aid, request_timeout)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            logger.warning("No entry found for arXiv ID %s", aid)
            continue
        # Extract metadata
        article_data[aid] = extract_metadata(entry, ns, aid)

    # Prepare a summary of the first few downloaded papers
    top_n = 3
    top_ids = list(article_data.keys())[:top_n]
    summary_lines: list[str] = []
    for i, aid in enumerate(top_ids):
        meta = article_data.get(aid, {})
        title = meta.get("Title", "N/A")
        pub_date = meta.get("Publication Date", "N/A")
        url = meta.get("URL", "")
        summary_lines.append(
            f"{i+1}. {title} ({pub_date}; arXiv ID: {aid}; URL: {url})"
        )

    summary = "\n".join(summary_lines)
    if len(article_data) > top_n:
        summary += f"\n...and {len(article_data) - top_n} more papers."

    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )
