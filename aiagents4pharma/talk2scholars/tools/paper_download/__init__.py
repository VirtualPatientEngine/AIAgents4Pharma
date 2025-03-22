#!/usr/bin/env python3
"""
This package provides modules for fetching and downloading academic papers from arXiv.
"""

# Import modules
from . import abstract_downloader
from . import arxiv_downloader
from . import download_arxiv_input
from .download_arxiv_input import download_arxiv_paper
from . import pubmed_downloader
from . import download_pubmed_input
from .download_pubmed_input import download_pubmed_paper
__all__ = [
    "abstract_downloader",
    "arxiv_downloader",
    "download_arxiv_input",
    "download_arxiv_paper",
    "pubmed_downloader",
    "download_pubmed_input",
    "download_pubmed_paper"
]
