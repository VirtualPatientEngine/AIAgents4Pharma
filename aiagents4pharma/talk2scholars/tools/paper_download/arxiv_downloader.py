"""
Arxiv Paper Downloader

This module provides an implementation of `AbstractPaperDownloader` for arXiv.
It connects to the arXiv API, retrieves metadata for a research paper, and
downloads the corresponding PDF.

By using an abstract base class, this implementation is extendable to other
APIs like PubMed, IEEE Xplore, etc.
"""
import xml.etree.ElementTree as ET
from typing import Any, Dict
import logging
import hydra
import requests
from .abstract_downloader import AbstractPaperDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivPaperDownloader(AbstractPaperDownloader):
    """
    Downloader class for arXiv.

    This class interfaces with the arXiv API to fetch metadata
    and retrieve PDFs of academic papers based on their arXiv IDs.
    """

    def __init__(self):
        """
        Initializes the arXiv paper downloader.

        Uses Hydra for configuration management to retrieve API details.
        """
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config",
                overrides=["tools/download_arxiv_paper=default"]
            )
            self.api_url = cfg.tools.download_arxiv_paper.api_url
            self.request_timeout = cfg.tools.download_arxiv_paper.request_timeout
            self.chunk_size = cfg.tools.download_arxiv_paper.chunk_size
            self.pdf_base_url = cfg.tools.download_arxiv_paper.pdf_base_url
    def fetch_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        Fetch metadata from arXiv for a given paper ID.

        Args:
            paper_id (str): The arXiv ID of the paper.

        Returns:
            Dict[str, Any]: A dictionary containing metadata, including the XML response.
        """
        logger.info("Fetching metadata from arXiv for paper ID: %s", paper_id)
        api_url = f"{self.api_url}?search_query=id:{paper_id}&start=0&max_results=1"
        response = requests.get(api_url, timeout=self.request_timeout)
        response.raise_for_status()
        return {"xml": response.text}

    def download_pdf(self, paper_id: str) -> Dict[str, Any]:
        """
        Download the PDF of a paper from arXiv.

        This function first retrieves the paper's metadata to locate the PDF link
        before downloading the file.

        Args:
            paper_id (str): The arXiv ID of the paper.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `pdf_object`: The binary content of the downloaded PDF.
                - `pdf_url`: The URL from which the PDF was fetched.
                - `arxiv_id`: The arXiv ID of the downloaded paper.
        """
        metadata = self.fetch_metadata(paper_id)

        # Parse the XML response to locate the PDF link.
        root = ET.fromstring(metadata["xml"])
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        pdf_url = next(
            (
                link.attrib.get("href")
                for entry in root.findall("atom:entry", ns)
                for link in entry.findall("atom:link", ns)
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )

        if not pdf_url:
            raise RuntimeError(f"Failed to download PDF for arXiv ID {paper_id}.")

        logger.info("Downloading PDF from: %s", pdf_url)
        pdf_response = requests.get(pdf_url, stream=True, timeout=self.request_timeout)
        pdf_response.raise_for_status()

        # Combine the PDF data from chunks.
        pdf_object = b"".join(
            chunk for chunk in pdf_response.iter_content(chunk_size=self.chunk_size) if chunk
            )

        return {
            "pdf_object": pdf_object,
            "pdf_url": pdf_url,
            "arxiv_id": paper_id,
        }
