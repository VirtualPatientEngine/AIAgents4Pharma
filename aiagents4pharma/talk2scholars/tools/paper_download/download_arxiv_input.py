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


# Helper to load arXiv download configuration
def _get_arxiv_config() -> Any:
    """Load arXiv download configuration."""
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_arxiv_paper=default"]
        )
    return cfg.tools.download_arxiv_paper


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


def _get_snippet(abstract: str) -> str:
    """Extract the first one or two sentences from an abstract."""
    if not abstract or abstract == "N/A":
        return ""
    sentences = abstract.split(". ")
    snippet_sentences = sentences[:2]
    snippet = ". ".join(snippet_sentences)
    if not snippet.endswith("."):
        snippet += "."
    return snippet


def _build_summary(article_data: dict[str, Any]) -> str:
    """Build a summary string for up to three papers with snippets."""
    top = list(article_data.values())[:3]
    lines: list[str] = []
    for idx, paper in enumerate(top):
        title = paper.get("Title", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        url = paper.get("URL", "")
        snippet = _get_snippet(paper.get("Abstract", ""))
        line = f"{idx+1}. {title} ({pub_date})"
        if url:
            line += f"\n   View PDF: {url}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)
    summary = "\n".join(lines)
    return (
        "Download was successful. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        "Top 3 papers:\n" + summary
    )


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
    cfg = _get_arxiv_config()
    api_url = cfg.api_url
    request_timeout = cfg.request_timeout

    # Aggregate results
    article_data: dict[str, Any] = {}
    for aid in arxiv_ids:
        logger.info("Processing arXiv ID: %s", aid)
        # Fetch and parse metadata
        entry = fetch_arxiv_metadata(api_url, aid, request_timeout).find(
            "atom:entry", {"atom": "http://www.w3.org/2005/Atom"}
        )
        if entry is None:
            logger.warning("No entry found for arXiv ID %s", aid)
            continue
        article_data[aid] = extract_metadata(
            entry, {"atom": "http://www.w3.org/2005/Atom"}, aid
        )

    # Build and return summary
    content = _build_summary(article_data)
    return Command(
        update={
            "article_data": article_data,
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=article_data,
                )
            ],
        }
    )
