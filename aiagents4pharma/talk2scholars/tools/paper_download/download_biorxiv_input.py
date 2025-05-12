#!/usr/bin/env python3
"""
Tool for downloading a bioRxiv paper PDF using its DOI segment.
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


class DownloadBioRxivPaperInput(BaseModel):
    """Input schema for the bioRxiv paper download tool."""
    biorxiv_suffix: str = Field(
        description="The bioRxiv suffix segment (e.g., '2025.05.07.652614v1') used to build the PDF URL."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

def fetch_biorxiv_metadata(
    api_url: str, biorxiv_suffix: str, request_timeout: int
) -> ET.Element:
    """Fetch and parse metadata from the bioRxiv API."""
    print("Inside fetch_biorxiv_metadata")
    query_url = f"{api_url}?search_query=id:{biorxiv_suffix}&start=0&max_results=1"
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return ET.fromstring(response.text)

def extract_biorxiv_metadata(biorxiv_suffix: str, api_url: str) -> dict:
    """
    Fetch metadata for a bioRxiv paper using its DOI.

    Args:
        doi (str): DOI of the paper, e.g., "10.1101/2025.05.07.652614v1"

    Returns:
        dict: Metadata including title, authors, abstract, publication date, and PDF URL
    """
    
    print("Inside extract_biorxiv_metadata")
    # if not doi.startswith("10.1101/"):
    #     raise ValueError("Only DOIs starting with '10.1101/' are supported.")

    # Construct API URL
    api_url = api_url + biorxiv_suffix
    print("API URL: ", api_url)
    # Call the API
    response = requests.get(api_url)
    
    print("Response: ", response)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch metadata. HTTP {response.status_code}: {api_url}")
    
    data = response.json()
    if not data.get("collection"):
        print("No metadata found for the provided DOI.")
        raise ValueError("No metadata found for the provided DOI.")

    # Extract fields
    entry = data["collection"][0]
    # doi_suffix = doi.split("10.1101/")[1]
    metadata = {
        "Title": entry.get("title", "N/A"),
        "Authors": entry.get("authors", "N/A"),
        "Abstract": entry.get("abstract", "N/A"),
        "Publication Date": entry.get("date", "N/A"),
        "URL": api_url,
        "pdf_url": f"https://www.biorxiv.org/content/{biorxiv_suffix}.full.pdf",
        "source": "biorxiv",
        "biorxiv_suffix": biorxiv_suffix
    }
    print("Metadata: ", metadata)
    return metadata

@tool(args_schema=DownloadBioRxivPaperInput, parse_docstring=True)
def download_biorxiv_paper(
    biorxiv_suffix: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Download a paper PDF from bioRxiv using its DOI segment (e.g., '2025.05.07.652614v1').
    """
    logger.info("Attempting to download bioRxiv PDF for ID: %s", biorxiv_suffix)


    # Construct the URL
    #{biorxiv_suffix}.full.pdf

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/pdf",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.biorxiv.org/",
        "Connection": "keep-alive"
    }

      # Load configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_arxiv_paper=default"]
        )
        api_url = cfg.tools.download_biorxiv_paper.api_url
        pdf_url = cfg.tools.download_biorxiv_paper.pdf_base_url
        api_url = api_url + biorxiv_suffix
        pdf_url = pdf_url + biorxiv_suffix + ".full.pdf"
        print("Hydra loaded successfully!")
        print("API URL: ", api_url)
        print(f"Hydra PDF URL: {pdf_url}")
        

    response = requests.get(pdf_url, headers=headers)
    
    print("Hydra Response: ", response)
    if response.status_code != 200:
        print("Response code is not 200")
        raise RuntimeError(f"Could not download PDF from {pdf_url}. Status code: {response.status_code}")

    filename = f"{biorxiv_suffix}.pdf"
    print("Filename: ", filename)

    metadata = extract_biorxiv_metadata(biorxiv_suffix, api_url)
    metadata["filename"] = filename

    article_data = {biorxiv_suffix: metadata}
    print("Article data: ", article_data)
    
    content = f"Successfully downloaded PDF for bioRxiv ID {biorxiv_suffix}"
    logger.info(content)
    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
