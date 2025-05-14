#!/usr/bin/env python3
"""
Tool for downloading medRxiv paper metadata and retrieving the PDF URL.
"""

import logging
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


class DownloadMedrxivPaperInput(BaseModel):
    """Input schema for the medRxiv paper download tool."""

    doi: str = Field(
        description="The medRxiv DOI used to retrieve the paper details and PDF URL."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

def fetch_medrxiv_metadata(doi: str) -> dict:
    """
    Fetch raw metadata JSON from medRxiv API for a given DOI.
    """
    api_url = f"https://api.biorxiv.org/details/medrxiv/{doi}"
    response = requests.get(api_url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch metadata. HTTP {response.status_code}")

    data = response.json()
    if not data.get("collection"):
        raise ValueError(f"No data found for DOI {doi}")
    return data["collection"][0]  # Return only the first item

def extract_metadata(paper: dict, doi: str) -> dict:
    """
    Extract relevant metadata fields from a medRxiv paper entry.
    """
    title = paper.get("title", "N/A")
    authors = paper.get("authors", "N/A")
    abstract = paper.get("abstract", "N/A")
    pub_date = paper.get("date", "N/A")
    doi_suffix = paper.get("doi", "").split("10.1101/")[-1]
    pdf_url = f"https://www.medrxiv.org/content/10.1101/{doi_suffix}.full.pdf"
    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{doi_suffix}.pdf",
        "source": "medrxiv",
        "medrxiv_id": doi
    }


@tool(args_schema=DownloadMedrxivPaperInput, parse_docstring=True)
def download_medrxiv_paper(
    doi: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URL for a medRxiv paper using its DOI.
    """
    logger.info("Fetching metadata from medRxiv for DOI: %s", doi)

    # Load configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_medrxiv_paper=default"]
        )
        api_url = cfg.tools.download_medrxiv_paper.api_url
        request_timeout = cfg.tools.download_medrxiv_paper.request_timeout

    print(f"API URL: {api_url}")
    print(f"Request Timeout: {request_timeout}")


    raw_data = fetch_medrxiv_metadata(doi)
    metadata = extract_metadata(raw_data, doi)
    article_data = {doi: metadata}

    content = f"Successfully retrieved metadata and PDF URL for medRxiv DOI {doi}"

    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
