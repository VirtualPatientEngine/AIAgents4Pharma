#!/usr/bin/env python3
"""
download_arxiv_paper: Tool for Fetching arXiv Papers and Downloading PDFs

This module connects to the arXiv API to retrieve metadata for a research paper and
download its corresponding PDF. It constructs an API query using a provided arXiv ID,
parses the XML response to locate the PDF link, and then downloads the PDF content.
The tool returns the PDF data along with metadata confirming the download operation.

By structuring this code with an abstract base class, it is future-proof for additional
APIs like PubMed or others. Right now, it only handles arXiv.
"""

import logging
from typing import Annotated, Any, Dict
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
import requests
import hydra


from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1️⃣ Abstract base class for future-proof design
class AbstractPaperDownloader(ABC):
    """
    Abstract base class for scholarly paper downloaders.
    This can be extended for different sources (e.g., arXiv, PubMed, IEEE Xplore, etc.).
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


# 2️⃣ Implementation of the abstract class specifically for arXiv
class ArxivPaperDownloader(AbstractPaperDownloader):
    """
    Downloader class for arXiv. It uses the arXiv API to retrieve metadata
    and downloads the paper PDF from the returned link.
    """

    def __init__(self):
        # Load Hydra configuration for the 'download_arxiv_paper' tool
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config",
                overrides=["tools/download_arxiv_paper=default"]
            )
            self.api_url = cfg.tools.download_arxiv_paper.api_url
            self.request_timeout = cfg.tools.download_arxiv_paper.request_timeout

    def fetch_metadata(self, paper_id: str) -> Dict[str, Any]:
        logger.info("Fetching metadata from arXiv for paper ID: %s", paper_id)
        api_url = f"{self.api_url}?search_query=id:{paper_id}&start=0&max_results=1"
        response = requests.get(api_url, timeout=self.request_timeout)
        response.raise_for_status()
        return {"xml": response.text}

    def download_pdf(self, paper_id: str) -> Dict[str, Any]:
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
            chunk for chunk in pdf_response.iter_content(chunk_size=1024) if chunk
            )

        return {
            "pdf_object": pdf_object,
            "pdf_url": pdf_url,
            "arxiv_id": paper_id,
        }


# 3️⃣ Input Schema remains the same (still specific to arXiv)
class DownloadArxivPaperInput(BaseModel):
    """
    Input schema for the arXiv paper download tool.

    Attributes:
        arxiv_id (str): The unique identifier of the paper on arXiv.
        tool_call_id (str): A unique identifier automatically injected to track this tool call.
    """
    arxiv_id: str = Field(
        description="The arXiv paper ID used to retrieve the paper details and PDF."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# 4️⃣ The actual tool function which leverages the ArxivPaperDownloader
@tool(args_schema=DownloadArxivPaperInput, parse_docstring=True)
def download_arxiv_paper(
    arxiv_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Download an arXiv paper's PDF using its unique arXiv ID.

    This function:
      1. Creates an ArxivPaperDownloader instance.
      2. Fetches metadata from arXiv using the provided `arxiv_id`.
      3. Downloads the PDF from the returned link.
      4. Returns a Command object containing the PDF data and a success message.
    """

    # Keep the same scoping approach for model_config


    downloader = ArxivPaperDownloader()
    pdf_data = downloader.download_pdf(arxiv_id)

    content = f"Successfully downloaded PDF for arXiv ID {arxiv_id}"

    return Command(
        update={
            "pdf_data": pdf_data,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )
