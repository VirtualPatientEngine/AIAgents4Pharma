#!/usr/bin/env python3
"""
Base Abstract Class definition to be followed when retrieving paper metadata from different sources.
"""
from abc import ABC, abstractmethod
from typing import Any,List

class BasePaperRetriever(ABC):
    """
    Abstract base class for implementing paper retrieval tools 
    (e.g., PubMed, arXiv, bioRxiv, medRxiv). 
    Defines a uniform interface for fetching and extracting metadata.
    """

    @abstractmethod
    def load_hydra_configs(self):
        """
        Load required configurations (e.g., URLs) using Hydra.
        """

    @abstractmethod
    def fetch_metadata(self,url: str, paper_id: str) -> dict:
        """
        Fetch metadata from the given URL using the paper ID.

        Args:
            url (str): The endpoint to fetch metadata from.
            paper_id (str): The unique identifier for the paper.

        Returns:
            dict: Dictionary containing the metadata.
        """

    @abstractmethod
    def extract_metadata(self,data: dict, paper_id: str) -> dict:
        """
        Extract the metadata fields from the XML root.

        Args:
            data (dict): The data from fetch_metadata.
            paper_id (str): The ID of the paper.

        Returns:
            dict: Dictionary containing extracted metadata.
        """

    @abstractmethod
    def paper_retriever(self, paper_ids: List[str]) -> dict[str, Any]:
        """
        Unified entry point for retrieving paper metadata and PDF.

        Args:
            paper_id List(str): Paper IDs (arXiv ID, DOIs, etc.).
            tool_call_id (str): Injected tool call ID for LangGraph communication.

        Returns:
            Dict: Dictionary containing metadata.
        """
