#!/usr/bin/env python3
"""
Tool for downloading MedRxiv paper metadata and retrieving the PDF URL.
"""

import logging
from typing import Any, List

import hydra
import requests

from .base_retreiver import BasePaperRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadMedrxivPaperInput(BasePaperRetriever):
    """Input schema for the MedRxiv paper download tool."""

    def __init__(self):
        self.ns = {"atom": "http://www.w3.org/2005/Atom"}
        self.request_timeout=None

    # Helper to load MedRxiv download configuration
    def load_hydra_configs(self) -> Any:
        """Load MedRxiv download configuration."""
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/download_medrxiv_paper=default"]
            )
        return cfg.tools.download_medrxiv_paper

    def fetch_metadata(
            self, url: str, paper_id: str
            ) -> dict:
        """
        Fetch metadata for a MedRxiv paper using its DOI and extract relevant fields.
        
        Parameters:
            doi (str): List of DOIs of the MedRxiv paper.
        
        Returns:
            dict: A dictionary containing the title, authors, abstract, publication date, and URLs.
        """
        # Strip any version suffix (e.g., v1) since MedRxiv's API is version-sensitive
        clean_doi = paper_id.split("v")[0]

        api_url = f"{url}{clean_doi}"
        logger.info("Fetching metadata from api url: %s", api_url)
        response = requests.get(api_url, timeout=self.request_timeout)
        response.raise_for_status()
        information = response.json()
        if not information["collection"]:
            raise ValueError(f"No metadata found for DOI: {clean_doi}")
        return information["collection"][0]

    def extract_metadata(self,data: dict,paper_id: str) -> dict:
        """
        Extract relevant metadata fields from a MedRxiv paper entry.
        """
        logger.info("Extracting metadata for %s",paper_id)
        title = data.get("title", "")
        authors = data.get("authors", "")
        abstract = data.get("abstract", "")
        pub_date = data.get("date", "")
        doi_suffix = data.get("doi", "").split("10.1101/")[-1]
        pdf_url = f"https://www.medrxiv.org/content/10.1101/{doi_suffix}.full.pdf"
        logger.info("PDF URL: %s", pdf_url)
        if requests.get(pdf_url,timeout=self.request_timeout).status_code != 200:
            print(f"No PDF found or access denied at {pdf_url}")
            return {}
        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
            "URL": pdf_url,
            "pdf_url": pdf_url,
            "filename": f"{doi_suffix}.pdf",
            "source": "medrxiv",
            "medrxiv_id": paper_id
        }

    def paper_retriever(
        self,
        paper_ids: List[str]
    ) -> dict[str, Any]:
        """
        Get metadata and PDF URLs for one or more MedRxiv papers using their unique dois.
        """
        logger.info("Fetching metadata from medrxiv for paper IDs: %s", paper_ids)

        # Load configuration
        config = self.load_hydra_configs()
        api_url = config.api_url
        self.request_timeout = config.request_timeout

        # Aggregate results
        article_data: dict[str, Any] = {}
        for doi in paper_ids:
            doi = doi.split(":")[1]
            logger.info("Processing DOI: %s", doi)
            # Fetch metadata
            entry = self.fetch_metadata(api_url,doi)
            # Extract relevant metadata
            metadata = self.extract_metadata(entry, doi)

            if metadata:
                article_data[doi]=metadata
                logger.info("Metadata fetched for %s",doi)
            else:
                logger.info("PDF could not be accessed")
                continue

        logger.info("Succesfully ran the biorxiv tool")
        return {
            "article_data": article_data
        }
