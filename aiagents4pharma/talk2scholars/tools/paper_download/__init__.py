#!/usr/bin/env python3
"""
This package provides modules for fetching and downloading academic papers from arXiv,
biorxiv and medrxiv.
"""

# Import modules
from . import arxiv_downloader
from . import base_paper_downloader
from . import medrxiv_downloader
from . import paper_downloader
from . import pubmed_downloader

__all__ = [
    "arxiv_downloader",
    "base_paper_downloader",
    "medrxiv_downloader",
    "paper_downloader",
    "pubmed_downloader",
]
