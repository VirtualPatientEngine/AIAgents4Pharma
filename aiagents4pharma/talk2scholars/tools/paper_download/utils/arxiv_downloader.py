#!/usr/bin/env python3
"""
ArXiv paper downloader implementation.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any

import requests

from .base_paper_downloader import BasePaperDownloader

logger = logging.getLogger(__name__)


class ArxivDownloader(BasePaperDownloader):
    """ArXiv-specific implementation of paper downloader."""

    def __init__(self, config: Any):
        """Initialize ArXiv downloader with configuration."""
        super().__init__(config)
        self.api_url = config.api_url
        self.pdf_base_url = config.pdf_base_url
        # XML namespace configuration
        self.xml_namespaces = getattr(
            config, "xml_namespace", {"atom": "http://www.w3.org/2005/Atom"}
        )

    def fetch_metadata(self, identifier: str) -> ET.Element:
        """
        Fetch paper metadata from arXiv API.

        Args:
            identifier: arXiv ID (e.g., '1234.5678' or '2301.12345')

        Returns:
            XML root element from arXiv API response

        Raises:
            requests.RequestException: If API call fails
            RuntimeError: If no entry found in response
        """
        query_url = f"{self.api_url}?search_query=id:{identifier}&start=0&max_results=1"
        logger.info("Fetching metadata for arXiv ID %s from: %s", identifier, query_url)

        response = requests.get(query_url, timeout=self.request_timeout)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        entry = root.find("atom:entry", self.xml_namespaces)

        if entry is None:
            raise RuntimeError("No entry found in arXiv API response")

        return root

    def construct_pdf_url(self, metadata: ET.Element, identifier: str) -> str:
        """
        Extract or construct PDF URL from arXiv metadata.

        Args:
            metadata: XML root from arXiv API
            identifier: arXiv ID

        Returns:
            PDF URL string
        """
        entry = metadata.find("atom:entry", self.xml_namespaces)

        if entry is None:
            return ""

        # Try to find PDF link in metadata first
        pdf_url = next(
            (
                link.attrib.get("href")
                for link in entry.findall("atom:link", self.xml_namespaces)
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )

        # Fallback to constructed PDF URL if not found in metadata
        if not pdf_url:
            pdf_url = f"{self.pdf_base_url}/{identifier}.pdf"
            logger.info("Using constructed PDF URL for %s: %s", identifier, pdf_url)

        return pdf_url

    def extract_paper_metadata(
        self,
        metadata: ET.Element,
        identifier: str,
        pdf_result: tuple[str, str] | None,
    ) -> dict[str, Any]:
        """
        Extract structured metadata from arXiv API response.

        Args:
            metadata: XML root from arXiv API
            identifier: arXiv ID
            pdf_result: Tuple of (temp_file_path, filename) if PDF downloaded

        Returns:
            Standardized paper metadata dictionary
        """
        entry = metadata.find("atom:entry", self.xml_namespaces)

        if entry is None:
            raise RuntimeError("No entry found in metadata")

        # Extract basic metadata
        basic_metadata = self._extract_basic_metadata(entry, self.xml_namespaces)

        # Handle PDF download results
        pdf_metadata = self._extract_pdf_metadata(pdf_result, identifier)

        # Combine all metadata
        return {
            **basic_metadata,
            **pdf_metadata,
            "source": "arxiv",
            "arxiv_id": identifier,
        }

    def _extract_basic_metadata(self, entry: ET.Element, ns: dict) -> dict[str, Any]:
        """Extract basic metadata (title, authors, abstract, date) from entry."""
        title = self._extract_title(entry, ns)
        authors = self._extract_authors(entry, ns)
        abstract = self._extract_abstract(entry, ns)
        pub_date = self._extract_publication_date(entry, ns)

        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
        }

    def _extract_title(self, entry: ET.Element, ns: dict) -> str:
        """Extract title from entry."""
        title_elem = entry.find("atom:title", ns)
        return (title_elem.text or "").strip() if title_elem is not None else "N/A"

    def _extract_authors(self, entry: ET.Element, ns: dict) -> list:
        """Extract authors from entry."""
        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_elem = author_elem.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        return authors

    def _extract_abstract(self, entry: ET.Element, ns: dict) -> str:
        """Extract abstract from entry."""
        summary_elem = entry.find("atom:summary", ns)
        return (summary_elem.text or "").strip() if summary_elem is not None else "N/A"

    def _extract_publication_date(self, entry: ET.Element, ns: dict) -> str:
        """Extract publication date from entry."""
        published_elem = entry.find("atom:published", ns)
        return (published_elem.text or "").strip() if published_elem is not None else "N/A"

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
        return "arXiv"

    def get_identifier_name(self) -> str:
        """Return identifier display name."""
        return "arXiv ID"

    def get_default_filename(self, identifier: str) -> str:
        """Generate default filename for arXiv paper."""
        return f"{identifier}.pdf"

    def _get_paper_identifier_info(self, paper: dict[str, Any]) -> str:
        """Get arXiv-specific identifier info for paper summary."""
        arxiv_id = paper.get("arxiv_id", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        return f" (arXiv:{arxiv_id}, {pub_date})"

    def _add_service_identifier(self, entry: dict[str, Any], identifier: str) -> None:
        """Add arXiv ID field to entry."""
        entry["arxiv_id"] = identifier
