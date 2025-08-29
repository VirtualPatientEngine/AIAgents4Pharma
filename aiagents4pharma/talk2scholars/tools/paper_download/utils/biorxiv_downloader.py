#!/usr/bin/env python3
"""
BioRxiv paper downloader implementation.
"""

import logging
import re
import tempfile
from typing import Any

import cloudscraper
import requests

from .base_paper_downloader import BasePaperDownloader

logger = logging.getLogger(__name__)


class BiorxivDownloader(BasePaperDownloader):
    """BioRxiv-specific implementation of paper downloader."""

    def __init__(self, config: Any):
        """Initialize BioRxiv downloader with configuration."""
        super().__init__(config)
        self.api_url = config.api_url
        self.pdf_base_url = getattr(
            config, "pdf_base_url", "https://www.biorxiv.org/content/10.1101/"
        )
        self.landing_url_template = getattr(
            config,
            "landing_url_template",
            "https://www.biorxiv.org/content/{doi}v{version}",
        )
        self.pdf_url_template = getattr(
            config,
            "pdf_url_template",
            "https://www.biorxiv.org/content/{doi}v{version}.full.pdf",
        )

        # Default values
        self.default_version = getattr(config, "default_version", "1")

        # CloudScraper specific settings
        self.cf_clearance_timeout = getattr(config, "cf_clearance_timeout", 30)
        self.session_reuse = getattr(config, "session_reuse", True)
        self.browser_config_type = getattr(config, "browser_config", {}).get("type", "custom")

        # Initialize shared CloudScraper session if enabled
        self._scraper = None
        if self.session_reuse:
            self._scraper = cloudscraper.create_scraper(
                browser={self.browser_config_type: self.user_agent},
                delay=self.cf_clearance_timeout,
            )

    def fetch_metadata(self, identifier: str) -> dict[str, Any]:
        """
        Fetch paper metadata from bioRxiv API.

        Args:
            identifier: DOI (e.g., '10.1101/2020.09.09.20191205')

        Returns:
            JSON response as dictionary from bioRxiv API

        Raises:
            requests.RequestException: If API call fails
            RuntimeError: If no collection data found in response
        """
        query_url = f"{self.api_url}/biorxiv/{identifier}/na/json"
        logger.info("Fetching metadata for DOI %s from: %s", identifier, query_url)

        # Use CloudScraper for metadata as well, in case API is behind CF protection
        scraper = self._scraper or cloudscraper.create_scraper(
            browser={self.browser_config_type: self.user_agent},
            delay=self.cf_clearance_timeout,
        )

        response = scraper.get(query_url, timeout=self.request_timeout)
        response.raise_for_status()

        paper_data = response.json()

        if "collection" not in paper_data or not paper_data["collection"]:
            raise RuntimeError("No collection data found in bioRxiv API response")

        return paper_data

    def construct_pdf_url(self, metadata: dict[str, Any], identifier: str) -> str:
        """
        Construct PDF URL from bioRxiv metadata and DOI.

        Args:
            metadata: JSON response from bioRxiv API
            identifier: DOI

        Returns:
            Constructed PDF URL string
        """
        if "collection" not in metadata or not metadata["collection"]:
            return ""

        paper = metadata["collection"][0]  # Get first (and should be only) paper
        version = paper.get("version", self.default_version)

        # Construct bioRxiv PDF URL using template
        pdf_url = self.pdf_url_template.format(doi=identifier, version=version)
        logger.info("Constructed PDF URL for DOI %s: %s", identifier, pdf_url)

        return pdf_url

    def download_pdf_to_temp(self, pdf_url: str, identifier: str) -> tuple[str, str] | None:
        """
        Override base method to use CloudScraper for bioRxiv PDF downloads.
        Includes landing page visit to handle CloudFlare protection.

        Args:
            pdf_url: URL to download PDF from
            identifier: DOI for logging

        Returns:
            Tuple of (temp_file_path, filename) or None if failed
        """
        if not pdf_url:
            logger.info("No PDF URL available for DOI %s", identifier)
            return None

        try:
            logger.info("Downloading PDF for DOI %s from %s", identifier, pdf_url)

            # Get scraper and visit landing page if needed
            scraper = self._get_scraper()
            self._visit_landing_page(scraper, pdf_url, identifier)

            # Download and save PDF
            response = scraper.get(pdf_url, timeout=self.request_timeout, stream=True)
            response.raise_for_status()

            temp_file_path = self._save_pdf_to_temp(response)
            filename = self._extract_filename(response, identifier)

            return temp_file_path, filename

        except requests.RequestException as e:
            logger.error("Failed to download PDF for DOI %s: %s", identifier, e)
            return None

    def _get_scraper(self):
        """Get or create CloudScraper instance."""
        return self._scraper or cloudscraper.create_scraper(
            browser={self.browser_config_type: self.user_agent},
            delay=self.cf_clearance_timeout,
        )

    def _visit_landing_page(self, scraper, pdf_url: str, identifier: str) -> None:
        """Visit landing page to handle CloudFlare protection."""
        if ".full.pdf" in pdf_url:
            landing_url = pdf_url.replace(".full.pdf", "")
            logger.info("Visiting landing page first: %s", landing_url)

            landing_response = scraper.get(landing_url, timeout=self.request_timeout)
            landing_response.raise_for_status()
            logger.info("Successfully accessed landing page for %s", identifier)

    def _save_pdf_to_temp(self, response) -> str:
        """Save PDF response to temporary file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    temp_file.write(chunk)
            temp_file_path = temp_file.name

        logger.info("BioRxiv PDF downloaded to temporary file: %s", temp_file_path)
        return temp_file_path

    def _extract_filename(self, response, identifier: str) -> str:
        """Extract filename from response headers or generate default."""
        filename = self.get_default_filename(identifier)

        content_disposition = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disposition:
            try:
                filename_match = re.search(
                    r'filename[*]?=(?:"([^"]+)"|([^;]+))', content_disposition
                )
                if filename_match:
                    extracted_filename = filename_match.group(1) or filename_match.group(2)
                    extracted_filename = extracted_filename.strip().strip('"')
                    if extracted_filename and extracted_filename.endswith(".pdf"):
                        filename = extracted_filename
                        logger.info("Extracted filename from header: %s", filename)
            except requests.RequestException as e:
                logger.warning("Failed to extract filename from header: %s", e)

        return filename

    def extract_paper_metadata(
        self,
        metadata: dict[str, Any],
        identifier: str,
        pdf_result: tuple[str, str] | None,
    ) -> dict[str, Any]:
        """
        Extract structured metadata from bioRxiv API response.

        Args:
            metadata: JSON response from bioRxiv API
            identifier: DOI
            pdf_result: Tuple of (temp_file_path, filename) if PDF downloaded

        Returns:
            Standardized paper metadata dictionary
        """
        if "collection" not in metadata or not metadata["collection"]:
            raise RuntimeError("No collection data found in metadata")

        paper = metadata["collection"][0]  # Get first (and should be only) paper

        # Extract basic metadata
        basic_metadata = self._extract_basic_metadata(paper, identifier)

        # Handle PDF download results
        pdf_metadata = self._extract_pdf_metadata(pdf_result, identifier)

        # Combine all metadata
        return {
            **basic_metadata,
            **pdf_metadata,
        }

    def _extract_basic_metadata(self, paper: dict[str, Any], identifier: str) -> dict[str, Any]:
        """Extract basic metadata from paper data."""
        # Extract basic fields
        title = paper.get("title", "N/A").strip()
        abstract = paper.get("abstract", "N/A").strip()
        pub_date = paper.get("date", "N/A").strip()
        category = paper.get("category", "N/A").strip()
        version = paper.get("version", "N/A")

        # Extract authors - typically in a semicolon-separated string
        authors = self._extract_authors(paper.get("authors", ""))

        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
            "DOI": identifier,
            "Category": category,
            "Version": version,
            "source": "biorxiv",
            "server": "biorxiv",
        }

    def _extract_authors(self, authors_str: str) -> list:
        """Extract and clean authors from semicolon-separated string."""
        if not authors_str:
            return []
        return [author.strip() for author in authors_str.split(";") if author.strip()]

    def _extract_pdf_metadata(
        self, pdf_result: tuple[str, str] | None, identifier: str
    ) -> dict[str, Any]:
        """Extract PDF-related metadata."""
        if pdf_result:
            temp_file_path, filename = pdf_result
            return {
                "URL": temp_file_path,
                "pdf_url": temp_file_path,
                "filename": filename,
                "access_type": "open_access_downloaded",
                "temp_file_path": temp_file_path,
            }

        return {
            "URL": "",
            "pdf_url": "",
            "filename": self.get_default_filename(identifier),
            "access_type": "download_failed",
            "temp_file_path": "",
        }

    def get_service_name(self) -> str:
        """Return service name."""
        return "bioRxiv"

    def get_identifier_name(self) -> str:
        """Return identifier display name."""
        return "DOI"

    def get_default_filename(self, identifier: str) -> str:
        """Generate default filename for bioRxiv paper."""
        # Sanitize DOI for filename use
        return f"{identifier.replace('/', '_').replace('.', '_')}.pdf"

    def _get_paper_identifier_info(self, paper: dict[str, Any]) -> str:
        """Get bioRxiv-specific identifier info for paper summary."""
        doi = paper.get("DOI", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        category = paper.get("Category", "N/A")

        info = f" (DOI:{doi}, {pub_date})"
        if category != "N/A":
            info += f"\n   Category: {category}"

        return info

    def _add_service_identifier(self, entry: dict[str, Any], identifier: str) -> None:
        """Add DOI and bioRxiv-specific fields to entry."""
        entry["DOI"] = identifier
        entry["Category"] = "N/A"
        entry["Version"] = "N/A"
        entry["server"] = "biorxiv"
