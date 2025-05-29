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

    doi: str = Field(description=
    """The medRxiv DOI, from search_helper or multi_helper or single_helper, 
    used to retrieve the paper details and PDF URL."""
    )
    logger.info("DOI Received: %s", doi)
    tool_call_id: Annotated[str, InjectedToolCallId]

# Fetching raw metadata from medRxiv API for a given DOI
def fetch_medrxiv_metadata(doi: str, api_url: str, request_timeout: int) -> dict:
    """
    Fetch metadata for a medRxiv paper using its DOI and extract relevant fields.

    Parameters:
        doi (str): The DOI of the medRxiv paper.

    Returns:
        dict: A dictionary containing the title, authors, abstract, publication date, and URLs.
    """
    # Strip any version suffix (e.g., v1) since bioRxiv's API is version-sensitive
    clean_doi = doi.split("v")[0]

    api_url = f"{api_url}{clean_doi}"
    logger.info("Fetching metadata from api url: %s", api_url)
    response = requests.get(api_url, timeout=request_timeout)
    response.raise_for_status()

    data = response.json()
    if not data.get("collection"):
        raise ValueError(f"No entry found for medRxiv ID {doi}")

    return data["collection"][0]

# Extracting relevant metadata fields from the raw data
def extract_metadata(paper: dict, doi: str) -> dict:
    """
    Extract relevant metadata fields from a medRxiv paper entry.
    """
    title = paper.get("title", "")
    authors = paper.get("authors", "")
    abstract = paper.get("abstract", "")
    pub_date = paper.get("date", "")
    doi_suffix = paper.get("doi", "").split("10.1101/")[-1]
    pdf_url = f"https://www.medrxiv.org/content/10.1101/{doi_suffix}.full.pdf"
    logger.info("PDF URL: %s", pdf_url)
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

# Tool to download medRxiv paper metadata and PDF URL
@tool(args_schema=DownloadMedrxivPaperInput, parse_docstring=True)
def download_medrxiv_paper(
    doi: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URL for a medRxiv paper using its doi or medrxiv id.
    """
    logger.info("Fetching metadata from medRxiv for DOI: %s", doi)

    # Load configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_medrxiv_paper=default"]
        )
        api_url = cfg.tools.download_medrxiv_paper.api_url
        request_timeout = cfg.tools.download_medrxiv_paper.request_timeout
        logger.info("API URL: %s", api_url)

    raw_data = fetch_medrxiv_metadata(doi, api_url, request_timeout)
    metadata = extract_metadata(raw_data, doi)
    article_data = {doi: metadata}

    content = f"Successfully retrieved metadata and PDF URL for medRxiv DOI {doi}"

    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
