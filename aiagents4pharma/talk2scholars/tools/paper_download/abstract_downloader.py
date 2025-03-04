"""
Abstract Base Class for Paper Downloaders.

This module defines the `AbstractPaperDownloader` class, which serves as a
base class for downloading scholarly papers from different sources
(e.g., arXiv, PubMed, IEEE Xplore). Any specific downloader should
inherit from this class and implement its methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class AbstractPaperDownloader(ABC):
    """
    Abstract base class for scholarly paper downloaders.

    This is designed to be extended for different paper sources
    like arXiv, PubMed, IEEE Xplore, etc. Each implementation
    must define methods for fetching metadata and downloading PDFs.
    """

    @abstractmethod
    def fetch_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        Fetch metadata for a given paper ID.

        Args:
            paper_id (str): The unique identifier for the paper.

        Returns:
            Dict[str, Any]: The metadata dictionary (format depends on the data source).
        """
        raise NotImplementedError

    @abstractmethod
    def download_pdf(self, paper_id: str) -> Dict[str, Any]:
        """
        Download the paper's PDF for a given paper ID.

        Args:
            paper_id (str): The unique identifier for the paper.

        Returns:
            Dict[str, Any]: Contains at least `pdf_object` and `pdf_url`.
        """
        raise NotImplementedError
