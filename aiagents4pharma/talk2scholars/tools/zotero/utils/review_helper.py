#!/usr/bin/env python3

"""
Utility for reviewing papers and saving them to Zotero.
"""

import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewData:
    """Helper class to organize review-related data."""

    def __init__(
        self,
        collection_path: str,
        fetched_papers: dict,
        tool_call_id: str,
        state: dict,
    ):
        self.collection_path = collection_path
        self.fetched_papers = fetched_papers
        self.tool_call_id = tool_call_id
        self.state = state
        self.total_papers = len(fetched_papers)
        self.papers_summary = self._create_papers_summary()
        self.papers_preview = "\n".join(self.papers_summary)
        self.review_info = self._create_review_info()

    def get_approval_message(self) -> str:
        """Get the formatted approval message for the review."""
        return (
            f"Human approved saving {self.total_papers} papers to Zotero "
            f"collection '{self.collection_path}'."
        )

    def get_custom_path_approval_message(self, custom_path: str) -> str:
        """Get the formatted approval message for a custom collection path."""
        return (
            f"Human approved saving papers to custom Zotero "
            f"collection '{custom_path}'."
        )

    def _create_papers_summary(self) -> List[str]:
        """Create a summary of papers for review."""
        summary = []
        for paper_id, paper in list(self.fetched_papers.items())[:5]:
            logger.info("Paper ID: %s", paper_id)
            title = paper.get("Title", "N/A")
            authors = ", ".join(
                [author.split(" (ID: ")[0] for author in paper.get("Authors", [])[:2]]
            )
            if len(paper.get("Authors", [])) > 2:
                authors += " et al."
            summary.append(f"- {title} by {authors}")

        if self.total_papers > 5:
            summary.append(f"... and {self.total_papers - 5} more papers")
        return summary

    def _create_review_info(self) -> dict:
        """Create the review information dictionary."""
        return {
            "action": "save_to_zotero",
            "collection_path": self.collection_path,
            "total_papers": self.total_papers,
            "papers_preview": self.papers_preview,
            "message": (
                f"Would you like to save {self.total_papers} papers to Zotero "
                f"collection '{self.collection_path}'? Please respond with a "
                f"structured decision using one of the following options: 'approve', "
                f"'reject', or 'custom' (with a custom_path)."
            ),
        }
