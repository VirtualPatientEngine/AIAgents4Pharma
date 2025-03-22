"""
PubMed Paper Downloader (Standalone Version)

Implements AbstractPaperDownloader without Hydra, for use in testing or script-based execution.
"""

import logging
from typing import Any, Dict
import hydra
import requests
from .abstract_downloader import AbstractPaperDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedPaperDownloader(AbstractPaperDownloader):
    """
    Downloader class for PubMed using static config (no Hydra).
    """

    def __init__(self):
        """
        Initializes the arXiv paper downloader.

        Uses Hydra for configuration management to retrieve API details.
        """
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/download_pubmed_paper=default"]
            )
        self.efetch_url = cfg.tools.download_pubmed_paper.efetch_url
        self.pdf_lookup_url = cfg.tools.download_pubmed_paper.pdf_lookup_url
        self.request_timeout = cfg.tools.download_pubmed_paper.request_timeout

    def fetch_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        Fetch metadata from PubMed.

        Args:
            paper_id (str): PubMed ID (PMID)

        Returns:
            Dict[str, Any]: Raw XML metadata
        """
        metadata_url = f"{self.efetch_url}?db=pubmed&id={paper_id}&retmode=xml"
        logger.info("Fetching metadata from: %s", metadata_url)

        response = requests.get(metadata_url, timeout=self.request_timeout)
        response.raise_for_status()
        return {"xml": response.text}

    def get_pmcid_from_pmid(self, pmid: str) -> str:
        """
        Map a PubMed ID (PMID) to a PubMed Central ID (PMCID) using ELink API.

        Returns:
            str: The PMCID (e.g., 'PMC12345678'), or None if not found.
        """
        logger.info("Mapping PMID %s to PMCID via ELink", pmid)
        elink_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
            f"?dbfrom=pubmed&linkname=pubmed_pmc&retmode=json&id={pmid}"
        )

        response = requests.get(elink_url, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        logger.info("elink url;%s", elink_url)
        try:
            pmcid = data["linksets"][0]["linksetdbs"][0]["links"][0]
            return pmcid
        except (KeyError, IndexError):
            return None

    def download_pdf(self, paper_id: str) -> Dict[str, Any]:
        """
        Download PDF using resolved PMCID (if available).
        """
        pmcid = self.get_pmcid_from_pmid(paper_id)
        if not pmcid:
            raise RuntimeError(f"Could not resolve PMCID for PMID {paper_id}")

        pdf_url = f"{self.pdf_lookup_url}{pmcid}/pdf"
        logger.info("Attempting to download PDF from: %s", pdf_url)
        headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                    }
        response = requests.get(pdf_url, stream=True, timeout=self.request_timeout, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"No PDF found or access denied at {pdf_url}")

        pdf_object = b"".join(chunk for chunk in response.iter_content(chunk_size=1024) if chunk)

        return {
            "pdf_object": pdf_object,
            "pdf_url": pdf_url,
            "pmid": paper_id,
            "pmcid": pmcid,
        }
