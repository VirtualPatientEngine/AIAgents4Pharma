#!/usr/bin/env python3
"""
MedRxiv paper downloader implementation.
"""

import logging
from typing import Any

import requests

from .base_paper_downloader import BasePaperDownloader

logger = logging.getLogger(__name__)


class MedrxivDownloader(BasePaperDownloader):
    """MedRxiv-specific implementation of paper downloader."""

    def __init__(self, config: Any):
        """Initialize MedRxiv downloader with configuration."""
        super().__init__(config)
        self.api_url = config.api_url
        self.pdf_url_template = getattr(
            config,
            "pdf_url_template",
            "https://www.medrxiv.org/content/{identifier}v{version}.full.pdf",
        )
        self.default_version = getattr(config, "default_version", "1")

    def fetch_metadata(self, identifier: str) -> dict[str, Any]:
        """
        Fetch paper metadata from medRxiv API.

        Args:
            identifier: DOI (e.g., '10.1101/2020.09.09.20191205')

        Returns:
            JSON response as dictionary from medRxiv API

        Raises:
            requests.RequestException: If API call fails
            RuntimeError: If no collection data found in response
        """
        query_url = f"{self.api_url}/medrxiv/{identifier}/na/json"
        logger.info("Fetching metadata for DOI %s from: %s", identifier, query_url)

        response = requests.get(query_url, timeout=self.request_timeout)
        response.raise_for_status()

        paper_data = response.json()

        if "collection" not in paper_data or not paper_data["collection"]:
            raise RuntimeError("No collection data found in medRxiv API response")

        return paper_data

    def construct_pdf_url(self, metadata: dict[str, Any], identifier: str) -> str:
        """
        Construct PDF URL from medRxiv metadata and DOI.

        Args:
            metadata: JSON response from medRxiv API
            identifier: DOI

        Returns:
            Constructed PDF URL string
        """
        if "collection" not in metadata or not metadata["collection"]:
            return ""

        paper = metadata["collection"][0]  # Get first (and should be only) paper
        version = paper.get("version", self.default_version)

        # Construct medRxiv PDF URL using template
        pdf_url = self.pdf_url_template.format(identifier=identifier, version=version)
        logger.info("Constructed PDF URL for DOI %s: %s", identifier, pdf_url)

        return pdf_url

    def extract_paper_metadata(
        self,
        metadata: dict[str, Any],
        identifier: str,
        pdf_result: tuple[str, str] | None,
    ) -> dict[str, Any]:
        """
        Extract structured metadata from medRxiv API response.

        Args:
            metadata: JSON response from medRxiv API
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
            "source": "medrxiv",
            "server": "medrxiv",
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
        return "medRxiv"

    def get_identifier_name(self) -> str:
        """Return identifier display name."""
        return "DOI"

    def get_default_filename(self, identifier: str) -> str:
        """Generate default filename for medRxiv paper."""
        # Sanitize DOI for filename use
        return f"{identifier.replace('/', '_').replace('.', '_')}.pdf"

    def _get_paper_identifier_info(self, paper: dict[str, Any]) -> str:
        """Get medRxiv-specific identifier info for paper summary."""
        doi = paper.get("DOI", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        category = paper.get("Category", "N/A")

        info = f" (DOI:{doi}, {pub_date})"
        if category != "N/A":
            info += f"\n   Category: {category}"

        return info

    def _add_service_identifier(self, entry: dict[str, Any], identifier: str) -> None:
        """Add DOI and medRxiv-specific fields to entry."""
        entry["DOI"] = identifier
        entry["Category"] = "N/A"
        entry["Version"] = "N/A"
        entry["server"] = "medrxiv"
