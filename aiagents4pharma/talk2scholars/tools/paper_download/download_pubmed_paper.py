#!/usr/bin/env python3
"""
Tool for downloading PubMedX paper metadata and retrieving the PDF URL.
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

def get_pubmedx_config():
    """Fetching Hydra configurations"""
    logger.info("Loading Hydra Configs:")
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_pubmed_paper=default"]
        )
        return (
            cfg.tools.download_pubmed_paper.metadata_url,
            cfg.tools.download_pubmed_paper.pdf_base_url,
            cfg.tools.download_pubmed_paper.map_url
        )
class DownloadPubMedXInput(BaseModel):
    """Input schema for the pubmedx paper download tool."""

    pmc_id: str = Field(
        description="The PMC ID used to retrieve the paper details and PDF URL."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

def map_ids(
        input_id: str,
        map_url: str,
        ) -> str:
    """Fetch the PMC ID of paper given other IDs for Pubmed retrieval"""
    response = requests.get(f"{map_url}?ids={input_id}", timeout=10)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    if((root.find('record') is not None) and (root.find('record').attrib.get('pmcid'))):
        logger.info("Retrieved PMC ID for the given id %s",input_id)
        return root.find('record').attrib.get('pmcid')
    raise RuntimeError(f"PMC id not found for {input_id}")

def fetch_metadata(
    metadata_url: str, pmc_id: str
) -> ET.Element:
    """Fetch and parse metadata from the pubmed API."""
    response = requests.get(
        metadata_url,
        params={
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml"
            },
        timeout=10
        )
    response.raise_for_status()
    return ET.fromstring(response.text)

def extract_metadata(root: ET.Element, pmc_id: str, pdf_download_url: str) -> dict:
    """Extract metadata for the PMC ID."""
    title_elem = root.find('.//article-title')
    title = title_elem.text if title_elem is not None else "N/A"

    # Abstract
    abstract_elem = root.find('.//abstract')
    abstract = ''.join(abstract_elem.itertext()).strip() if abstract_elem is not None else "N/A"

    # Authors
    authors = ', '.join(
        f"{name.findtext('given-names', default='')} {name.findtext('surname', default='')}".strip()
        for contrib in root.findall('.//contrib[@contrib-type="author"]')
        if (name := contrib.find('name')) is not None
    ) or "N/A"

    # Publication Data
    pub_date_elem = root.find('.//published')
    pub_date = pub_date_elem.text.strip() if pub_date_elem is not None else "N/A"

    pdf_url = f"{pdf_download_url}{pmc_id}?pdf=render"
    if requests.get(pdf_url,timeout=10).status_code != 200:
        raise RuntimeError(f"No PDF found or access denied at {pdf_url}")

    logger.info("Metadata from PubMedx for paper PMC ID: %s", pmc_id)
    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{pmc_id}.pdf",
        "source": "pubmed",
        "pmc_id": pmc_id,
    }


@tool(args_schema=DownloadPubMedXInput, parse_docstring=True)
def download_pubmedx_paper(
    pmc_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URL for an PubMedX paper using unique DOI.
    """
    logger.info("Fetching metadata from PubMedx for paper PMC ID: %s", pmc_id)

    metadata_url,pdf_download_url,map_url = get_pubmedx_config()

    #Mapping given id to pmc_id
    if pmc_id.lower().startswith("pmc"):
        pass
    else:
        pmc_id = map_ids(pmc_id,map_url)
    # Extract metadata
    raw_data = fetch_metadata(metadata_url,pmc_id)
    metadata = extract_metadata(raw_data,pmc_id,pdf_download_url)

    # Create article_data entry with the paper ID as the key
    article_data = {pmc_id: metadata}
    content = f"Successfully retrieved metadata for PMC ID {pmc_id}"

    return Command(
        update={
            "article_data": article_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
