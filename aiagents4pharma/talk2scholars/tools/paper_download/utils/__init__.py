#!/usr/bin/env python3
"""
This package provides modules for fetching and downloading academic papers from arXiv,
biorxiv and medrxiv.
"""

# Import modules
from . import (
    arxiv_downloader,
    base_paper_downloader,
    biorxiv_downloader,
    medrxiv_downloader,
    pubmed_downloader,
)

__all__ = [
    "arxiv_downloader",
    "base_paper_downloader",
    "biorxiv_downloader",
    "medrxiv_downloader",
    "pubmed_downloader",
]
