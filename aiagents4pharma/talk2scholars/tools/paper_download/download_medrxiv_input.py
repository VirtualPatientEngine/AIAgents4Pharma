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

# Fetching raw metadata from medRxiv API for a given DOI
def fetch_medrxiv_metadata(doi: str) -> dict:
    """
    Fetch metadata JSON from medRxiv API for a given DOI.
    """
    # Strip any version suffix (e.g., v1) since bioRxiv's API is version-sensitive
    clean_doi = doi.split("v")[0]

    base_url = "https://api.biorxiv.org/details/medrxiv/"
    api_url = f"{base_url}{clean_doi}"
    response = requests.get(api_url, timeout=10)

    data = response.json()
    if not data.get("collection"):
        raise ValueError(f"No entry found for medRxiv ID {doi}")

    return data["collection"][0]

# Extracting relevant metadata fields from the raw data
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
        logger.info("API URL: %s", api_url)
        logger.info("Request Timeout: %s", request_timeout)


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
