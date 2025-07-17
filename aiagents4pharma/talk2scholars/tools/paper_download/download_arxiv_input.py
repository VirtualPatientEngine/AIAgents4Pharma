#!/usr/bin/env python3
"""
Tool for downloading arXiv paper metadata and retrieving the PDF URL.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, List
import hydra
import requests

from .base_retreiver import BasePaperRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadArxivPaperInput(BasePaperRetriever):
    """Input schema for the arXiv paper download tool."""

    def __init__(self):
        self.ns = {"atom": "http://www.w3.org/2005/Atom"}
        self.request_timeout = None

    # Helper to load arXiv download configuration
    def load_hydra_configs(self) -> Any:
        """Load arXiv download configuration."""
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/download_arxiv_paper=default"]
            )
        return cfg.tools.download_arxiv_paper


    def fetch_metadata(
        self,url: str, paper_id: str
        ) -> dict:
        """Fetch and parse metadata from the arXiv API."""
        query_url = f"{url}?search_query=id:{paper_id}&start=0&max_results=1"
        response = requests.get(query_url, timeout=self.request_timeout)
        response.raise_for_status()
        return {"data": ET.fromstring(response.text)}


    def extract_metadata(self,data: dict,paper_id: str) -> dict:
        """Extract metadata from the XML xml_root."""
        xml_root = data["data"].find(
                "atom:entry", self.ns
            )
        title_elem = xml_root.find("atom:title", self.ns)
        title = (title_elem.text or "").strip() if title_elem is not None else "N/A"

        authors = []
        for author_elem in xml_root.findall("atom:author", self.ns):
            name_elem = author_elem.find("atom:name", self.ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        summary_elem = xml_root.find("atom:summary", self.ns)
        abstract = (summary_elem.text or "").strip() if summary_elem is not None else "N/A"

        published_elem = xml_root.find("atom:published", self.ns)
        pub_date = (
            (published_elem.text or "").strip() if published_elem is not None else "N/A"
        )

        pdf_url = next(
            (
                link.attrib.get("href")
                for link in xml_root.findall("atom:link", self.ns)
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )
        if not pdf_url:
            raise RuntimeError(f"Could not find PDF URL for arXiv ID {paper_id}")
        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
            "URL": pdf_url,
            "pdf_url": pdf_url,
            "filename": f"{paper_id}.pdf",
            "source": "arxiv",
            "arxiv_id": paper_id,
        }

    def paper_retriever(
        self,
        paper_ids: List[str]
    ) -> dict[str, Any]:
        """
        Get metadata and PDF URLs for one or more arXiv papers using their unique arXiv IDs.
        """
        logger.info("Fetching metadata from arXiv for paper IDs: %s", paper_ids)

        # Load configuration
        cfg = self.load_hydra_configs()
        api_url = cfg.api_url
        self.request_timeout = cfg.request_timeout

        # Aggregate results
        article_data: dict[str, Any] = {}
        for aid in paper_ids:
            aid = aid.split(":")[1]
            logger.info("Processing arXiv ID: %s", aid)
            # Fetch and parse metadata
            xml_root = self.fetch_metadata(api_url, aid)
            if xml_root["data"].find(
                "atom:entry", self.ns
            ) is None:
                logger.warning("No xml_root found for arXiv ID %s", aid)
                continue
            article_data[aid] = self.extract_metadata(
                xml_root, aid
            )
            logger.info("Successfully fetched details for %s",aid)
        # Build and return summary
        return {
            "article_data": article_data
        }
