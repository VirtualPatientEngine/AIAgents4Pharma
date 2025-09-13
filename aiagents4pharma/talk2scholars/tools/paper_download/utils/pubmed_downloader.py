#!/usr/bin/env python3
"""
PubMed paper downloader implementation.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, cast

import requests
from bs4 import BeautifulSoup, Tag

from .base_paper_downloader import BasePaperDownloader

logger = logging.getLogger(__name__)


class PubmedDownloader(BasePaperDownloader):
    """PubMed-specific implementation of paper downloader."""

    def __init__(self, config: Any):
        """Initialize PubMed downloader with configuration."""
        super().__init__(config)
        self.id_converter_url = config.id_converter_url
        self.oa_api_url = config.oa_api_url

        # Alternative PDF sources
        self.europe_pmc_base_url = config.europe_pmc_base_url
        self.pmc_page_base_url = config.pmc_page_base_url
        self.direct_pmc_pdf_base_url = config.direct_pmc_pdf_base_url

        # URL conversion for NCBI FTP links
        self.ftp_base_url = config.ftp_base_url
        self.https_base_url = config.https_base_url
        # Configuration values
        self.id_converter_format = getattr(config, "id_converter_format", "json")
        self.pdf_meta_name = getattr(config, "pdf_meta_name", "citation_pdf_url")
        self.default_error_code = getattr(config, "default_error_code", "unknown")

    def fetch_metadata(self, identifier: str) -> dict[str, Any]:
        """
        Fetch paper metadata from PubMed ID Converter API.

        Args:
            identifier: PMID (e.g., '12345678')

        Returns:
            JSON response from PMC ID Converter API

        Raises:
            requests.RequestException: If API call fails
            RuntimeError: If no records found in response
        """
        query_url = f"{self.id_converter_url}?ids={identifier}&format={self.id_converter_format}"
        logger.info("Fetching metadata from ID converter for PMID %s: %s", identifier, query_url)

        response = requests.get(query_url, timeout=self.request_timeout)
        response.raise_for_status()

        result = response.json()
        logger.info("ID converter response for PMID %s: %s", identifier, result)

        if "records" not in result or not result["records"]:
            raise RuntimeError("No records found in PMC ID Converter API response")

        return result

    def construct_pdf_url(self, metadata: dict[str, Any], identifier: str) -> str:
        """
        Construct PDF URL using multiple fallback strategies.

        Args:
            metadata: JSON response from ID converter
            identifier: PMID

        Returns:
            PDF URL string (empty if no PDF available)
        """
        if "records" not in metadata or not metadata["records"]:
            return ""

        record = metadata["records"][0]
        pmcid = record.get("pmcid", "")

        if not pmcid or pmcid == "N/A":
            logger.info("No PMCID available for PDF fetch: PMID %s", identifier)
            return ""

        return self._fetch_pdf_url_with_fallbacks(pmcid)

    def _fetch_pdf_url_with_fallbacks(self, pmcid: str) -> str:
        """
        Fetch PDF URL from OA API with comprehensive fallback strategies.

        Args:
            pmcid: PMC ID (e.g., 'PMC1234567')

        Returns:
            PDF URL string (empty if all strategies fail)
        """
        logger.info("Fetching PDF URL for PMCID: %s", pmcid)

        # Strategy 1: Official OA API (fastest when it works)
        pdf_url = self._try_oa_api(pmcid)
        if pdf_url:
            return pdf_url

        # Strategy 2: Europe PMC Service (most reliable fallback)
        pdf_url = self._try_europe_pmc(pmcid)
        if pdf_url:
            return pdf_url

        # Strategy 3: Scrape PMC page for citation_pdf_url meta tag
        pdf_url = self._try_pmc_page_scraping(pmcid)
        if pdf_url:
            return pdf_url

        # Strategy 4: Direct PMC PDF URL pattern (least reliable)
        pdf_url = self._try_direct_pmc_url(pmcid)
        if pdf_url:
            return pdf_url

        logger.warning("All PDF URL strategies failed for PMCID: %s", pmcid)
        return ""

    def _try_oa_api(self, pmcid: str) -> str:
        """Try to get PDF URL from official OA API."""
        query_url = f"{self.oa_api_url}?id={pmcid}"
        logger.info("Trying OA API for PMCID %s: %s", pmcid, query_url)

        try:
            response = requests.get(query_url, timeout=self.request_timeout)
            response.raise_for_status()

            logger.info("OA API response for PMCID %s: %s", pmcid, response.text[:500])

            # Parse XML response

            root = ET.fromstring(response.text)

            # Check for error first
            error_elem = root.find(".//error")
            if error_elem is not None:
                error_code = error_elem.get("code", self.default_error_code)
                error_text = error_elem.text or "unknown error"
                logger.info("OA API error for PMCID %s: %s - %s", pmcid, error_code, error_text)
                return ""

            # Look for PDF link
            pdf_link = root.find(".//link[@format='pdf']")
            if pdf_link is not None:
                pdf_url = pdf_link.get("href", "")
                logger.info("Found PDF URL from OA API for PMCID %s: %s", pmcid, pdf_url)

                # Convert FTP links to HTTPS for download compatibility
                if pdf_url.startswith(self.ftp_base_url):
                    pdf_url = pdf_url.replace(self.ftp_base_url, self.https_base_url)
                    logger.info("Converted FTP to HTTPS for %s: %s", pmcid, pdf_url)

                return pdf_url

        except requests.RequestException as e:
            logger.info("OA API failed for %s: %s", pmcid, str(e))

        return ""

    def _try_europe_pmc(self, pmcid: str) -> str:
        """Try Europe PMC service for PDF."""
        europe_pmc_url = f"{self.europe_pmc_base_url}?accid={pmcid}&blobtype=pdf"
        logger.info("Trying Europe PMC service for %s: %s", pmcid, europe_pmc_url)

        try:
            response = requests.head(europe_pmc_url, timeout=self.request_timeout)
            if response.status_code == 200:
                logger.info("Europe PMC service works for %s", pmcid)
                return europe_pmc_url
        except requests.RequestException as e:
            logger.info("Europe PMC service failed for %s: %s", pmcid, str(e))

        return ""

    def _try_pmc_page_scraping(self, pmcid: str) -> str:
        """Try scraping PMC page for PDF meta tag."""
        pmc_page_url = f"{self.pmc_page_base_url}/{pmcid}/"
        logger.info("Scraping PMC page for PDF meta tag for %s: %s", pmcid, pmc_page_url)

        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(pmc_page_url, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Look for PDF meta tag
            pdf_meta = soup.find("meta", attrs={"name": self.pdf_meta_name})
            if pdf_meta is not None:
                # Cast to Tag to help type checker understand this is a BeautifulSoup Tag object
                meta_tag = cast(Tag, pdf_meta)
                content = meta_tag.get("content")
                if content:
                    logger.info(
                        "Found %s meta tag for %s: %s",
                        self.pdf_meta_name,
                        pmcid,
                        content,
                    )
                    return str(content)

        except requests.RequestException as e:
            logger.info("PMC page scraping failed for %s: %s", pmcid, str(e))

        return ""

    def _try_direct_pmc_url(self, pmcid: str) -> str:
        """Try direct PMC PDF URL pattern."""
        direct_pmc_url = f"{self.direct_pmc_pdf_base_url}/{pmcid}/pdf/"
        logger.info("Trying direct PMC PDF URL for %s: %s", pmcid, direct_pmc_url)

        try:
            response = requests.head(direct_pmc_url, timeout=self.request_timeout)
            if response.status_code == 200:
                logger.info("Direct PMC PDF URL works for %s", pmcid)
                return direct_pmc_url
        except requests.RequestException as e:
            logger.info("Direct PMC PDF URL failed for %s: %s", pmcid, str(e))

        return ""

    def extract_paper_metadata(
        self,
        metadata: dict[str, Any],
        identifier: str,
        pdf_result: tuple[str, str] | None,
    ) -> dict[str, Any]:
        """
        Extract structured metadata from PubMed ID converter response.

        Args:
            metadata: JSON response from ID converter
            identifier: PMID
            pdf_result: Tuple of (temp_file_path, filename) if PDF downloaded

        Returns:
            Standardized paper metadata dictionary
        """
        if "records" not in metadata or not metadata["records"]:
            raise RuntimeError("No records found in metadata")

        record = metadata["records"][0]  # Get first (and should be only) record

        # Extract basic fields from ID converter
        pmcid = record.get("pmcid", "N/A")
        doi = record.get("doi", "N/A")

        # Handle PDF download results
        if pdf_result:
            temp_file_path, filename = pdf_result
            access_type = "open_access_downloaded"
            pdf_url = temp_file_path  # Use local temp file path
        else:
            temp_file_path = ""
            filename = self.get_default_filename(identifier)
            access_type = "abstract_only" if pmcid != "N/A" else "no_pmcid"
            pdf_url = ""

        # Note: For PubMed, we don't get title/authors from ID converter
        # In a real implementation, you might want to call E-utilities for full metadata
        # For now, we'll use placeholders and focus on the ID conversion functionality

        return {
            "Title": (
                f"PubMed Article {identifier}"
            ),  # Placeholder - would need E-utilities for real title
            "Authors": [],  # Placeholder - would need E-utilities for real authors
            "Abstract": "Abstract available in PubMed",  # Placeholder
            "Publication Date": "N/A",  # Would need E-utilities for this
            "PMID": identifier,
            "PMCID": pmcid,
            "DOI": doi,
            "Journal": "N/A",  # Would need E-utilities for this
            "URL": pdf_url,
            "pdf_url": pdf_url,
            "access_type": access_type,
            "filename": filename,
            "source": "pubmed",
            "temp_file_path": temp_file_path,
        }

    def get_service_name(self) -> str:
        """Return service name."""
        return "PubMed"

    def get_identifier_name(self) -> str:
        """Return identifier display name."""
        return "PMID"

    def get_default_filename(self, identifier: str) -> str:
        """Generate default filename for PubMed paper."""
        return f"pmid_{identifier}.pdf"

    def get_snippet(self, abstract: str) -> str:
        """Override to handle PubMed-specific abstract placeholder."""
        if not abstract or abstract == "N/A" or abstract == "Abstract available in PubMed":
            return ""
        return super().get_snippet(abstract)

    def _get_paper_identifier_info(self, paper: dict[str, Any]) -> str:
        """Get PubMed-specific identifier info for paper summary."""
        pmid = paper.get("PMID", "N/A")
        pmcid = paper.get("PMCID", "N/A")

        info = f" (PMID: {pmid})"
        if pmcid != "N/A":
            info += f"\n   PMCID: {pmcid}"

        return info

    def _add_service_identifier(self, entry: dict[str, Any], identifier: str) -> None:
        """Add PMID and PubMed-specific fields to entry."""
        entry["PMID"] = identifier
        entry["PMCID"] = "N/A"
        entry["DOI"] = "N/A"
        entry["Journal"] = "N/A"
