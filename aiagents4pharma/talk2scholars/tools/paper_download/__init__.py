#!/usr/bin/env python3
"""
This package provides modules for fetching and downloading academic papers from arXiv and pubmed.
"""

# Import modules
from . import download_arxiv_input,download_pubmed_paper

__all__ = [
    "download_arxiv_input",
    "download_pubmed_paper",
]
