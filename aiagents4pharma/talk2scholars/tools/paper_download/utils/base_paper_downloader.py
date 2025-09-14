#!/usr/bin/env python3
"""
Abstract base class for paper download tools.
Provides common functionality for arXiv, medRxiv, PubMed, and future paper sources.
"""

import logging
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import requests

# Configure logging
logger = logging.getLogger(__name__)


class BasePaperDownloader(ABC):
    """Abstract base class for paper download tools."""

    def __init__(self, config: Any):
        """Initialize with service-specific configuration."""
        self.config = config
        self.request_timeout = getattr(config, "request_timeout", 15)
        self.chunk_size = getattr(config, "chunk_size", 8192)
        self.user_agent = getattr(
            config, "user_agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        )

    # Abstract methods that each service must implement
    @abstractmethod
    def fetch_metadata(self, identifier: str) -> Any:
        """
        Fetch paper metadata from the service API.

        Args:
            identifier: Paper identifier (arXiv ID, DOI, PMID, etc.)

        Returns:
            Service-specific metadata object (XML, JSON, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def construct_pdf_url(self, metadata: Any, identifier: str) -> str:
        """
        Construct or extract PDF URL from metadata.

        Args:
            metadata: Metadata returned from fetch_metadata()
            identifier: Original paper identifier

        Returns:
            PDF URL string (empty if not available)
        """
        raise NotImplementedError

    @abstractmethod
    def extract_paper_metadata(
        self, metadata: Any, identifier: str, pdf_result: tuple[str, str] | None
    ) -> dict[str, Any]:
        """
        Extract and structure metadata into standardized format.

        Args:
            metadata: Raw metadata from API
            identifier: Original paper identifier
            pdf_result: Tuple of (temp_file_path, filename) if PDF downloaded

        Returns:
            Standardized paper metadata dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def get_service_name(self) -> str:
        """Return service name (e.g., 'arxiv', 'medrxiv', 'pubmed')."""
        raise NotImplementedError

    @abstractmethod
    def get_identifier_name(self) -> str:
        """Return identifier display name (e.g., 'arXiv ID', 'DOI', 'PMID')."""
        raise NotImplementedError

    @abstractmethod
    def get_default_filename(self, identifier: str) -> str:
        """Generate default filename for the paper PDF."""
        raise NotImplementedError

    # Common methods shared by all services
    def download_pdf_to_temp(self, pdf_url: str, identifier: str) -> tuple[str, str] | None:
        """
        Download PDF from URL to a temporary file.

        Args:
            pdf_url: URL to download PDF from
            identifier: Paper identifier for logging

        Returns:
            Tuple of (temp_file_path, filename) or None if failed
        """
        if not pdf_url:
            logger.info("No PDF URL available for %s %s", self.get_identifier_name(), identifier)
            return None

        try:
            logger.info(
                "Downloading PDF for %s %s from %s",
                self.get_identifier_name(),
                identifier,
                pdf_url,
            )

            headers = {"User-Agent": self.user_agent}
            response = requests.get(
                pdf_url, headers=headers, timeout=self.request_timeout, stream=True
            )
            response.raise_for_status()

            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        temp_file.write(chunk)
                temp_file_path = temp_file.name

            logger.info(
                "%s PDF downloaded to temporary file: %s",
                self.get_service_name(),
                temp_file_path,
            )

            # Try to extract filename from Content-Disposition header
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

            return temp_file_path, filename

        except (requests.exceptions.RequestException, OSError) as e:
            logger.error(
                "Failed to download PDF for %s %s: %s",
                self.get_identifier_name(),
                identifier,
                e,
            )
            return None

    def get_snippet(self, abstract: str) -> str:
        """
        Extract the first one or two sentences from an abstract.

        Args:
            abstract: Full abstract text

        Returns:
            Snippet of first 1-2 sentences
        """
        if not abstract or abstract == "N/A":
            return ""

        sentences = abstract.split(". ")
        snippet_sentences = sentences[:2]
        snippet = ". ".join(snippet_sentences)

        if not snippet.endswith("."):
            snippet += "."

        return snippet

    def create_error_entry(self, identifier: str, error_msg: str) -> dict[str, Any]:
        """
        Create standardized error entry for failed paper processing.

        Args:
            identifier: Paper identifier
            error_msg: Error message

        Returns:
            Error entry dictionary
        """
        return {
            "Title": "Error fetching paper",
            "Authors": [],
            "Abstract": f"Error: {error_msg}",
            "Publication Date": "N/A",
            "URL": "",
            "pdf_url": "",
            "filename": self.get_default_filename(identifier),
            "source": self.get_service_name(),
            "access_type": "error",
            "temp_file_path": "",
            "error": error_msg,
            # Service-specific identifier field will be added by subclasses
        }

    def build_summary(self, article_data: dict[str, Any]) -> str:
        """
        Build a summary string for up to three papers with snippets.

        Args:
            article_data: Dictionary of paper data keyed by identifier

        Returns:
            Formatted summary string
        """
        top = list(article_data.values())[:3]
        lines: list[str] = []
        downloaded_count = sum(
            1
            for paper in article_data.values()
            if paper.get("access_type") == "open_access_downloaded"
        )

        for idx, paper in enumerate(top):
            title = paper.get("Title", "N/A")
            access_type = paper.get("access_type", "N/A")
            temp_file_path = paper.get("temp_file_path", "")
            snippet = self.get_snippet(paper.get("Abstract", ""))

            # Build paper line with service-specific identifier info
            line = f"{idx + 1}. {title}"
            line += self._get_paper_identifier_info(paper)
            line += f"\n   Access: {access_type}"

            if temp_file_path:
                line += f"\n   Downloaded to: {temp_file_path}"
            if snippet:
                line += f"\n   Abstract snippet: {snippet}"

            lines.append(line)

        summary = "\n".join(lines)
        service_name = self.get_service_name()

        return (
            f"Download was successful from {service_name}. "
            "Papers metadata are attached as an artifact. "
            "Here is a summary of the results:\n"
            f"Number of papers found: {len(article_data)}\n"
            f"PDFs successfully downloaded: {downloaded_count}\n"
            "Top 3 papers:\n" + summary
        )

    @abstractmethod
    def _get_paper_identifier_info(self, paper: dict[str, Any]) -> str:
        """
        Get service-specific identifier info for paper summary.

        Args:
            paper: Paper metadata dictionary

        Returns:
            Formatted identifier string (e.g., " (arXiv:1234.5678, 2023-01-01)")
        """
        raise NotImplementedError

    def process_identifiers(self, identifiers: list[str]) -> dict[str, Any]:
        """
        Main processing loop for downloading papers.

        Args:
            identifiers: List of paper identifiers

        Returns:
            Dictionary of paper data keyed by identifier
        """
        logger.info(
            "Processing %d identifiers from %s: %s",
            len(identifiers),
            self.get_service_name(),
            identifiers,
        )

        article_data: dict[str, Any] = {}

        for identifier in identifiers:
            logger.info("Processing %s: %s", self.get_identifier_name(), identifier)

            try:
                # Step 1: Fetch metadata
                metadata = self.fetch_metadata(identifier)

                # Step 2: Extract PDF URL
                pdf_url = self.construct_pdf_url(metadata, identifier)

                # Step 3: Download PDF if available
                pdf_result = None
                if pdf_url:
                    pdf_result = self.download_pdf_to_temp(pdf_url, identifier)

                # Step 4: Extract and structure metadata
                article_data[identifier] = self.extract_paper_metadata(
                    metadata, identifier, pdf_result
                )

            except requests.RequestException as e:
                logger.warning(
                    "Error processing %s %s: %s",
                    self.get_identifier_name(),
                    identifier,
                    str(e),
                )

                # Create error entry
                error_entry = self.create_error_entry(identifier, str(e))
                # Add service-specific identifier field
                self._add_service_identifier(error_entry, identifier)
                article_data[identifier] = error_entry

        return article_data

    @abstractmethod
    def _add_service_identifier(self, entry: dict[str, Any], identifier: str) -> None:
        """
        Add service-specific identifier field to entry.

        Args:
            entry: Paper entry dictionary to modify
            identifier: Original identifier
        """
        raise NotImplementedError
