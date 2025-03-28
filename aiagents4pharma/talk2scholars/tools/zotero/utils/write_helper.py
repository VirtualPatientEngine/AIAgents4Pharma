#!/usr/bin/env python3

"""
Utility for zotero write tool.
"""

import logging
from typing import Any, Dict
import hydra
from pyzotero import zotero
from .zotero_path import (
    find_or_create_collection,
    fetch_papers_for_save,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroWriteData:
    """Helper class to organize Zotero write-related data."""

    def __init__(
        self,
        tool_call_id: str,
        collection_path: str,
        state: dict,
    ):
        self.tool_call_id = tool_call_id
        self.collection_path = collection_path
        self.state = state
        self.cfg = self._load_config()
        self.zot = self._init_zotero_client()
        self.fetched_papers = fetch_papers_for_save(state)
        self.normalized_path = collection_path.rstrip("/").lower()
        self.zotero_items = []
        self.content = ""

    def _load_config(self) -> Any:
        """Load hydra configuration."""
        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/zotero_write=default"]
            )
            logger.info("Loaded configuration for Zotero write tool")
            return cfg.tools.zotero_write

    def _init_zotero_client(self) -> zotero.Zotero:
        """Initialize Zotero client."""
        logger.info(
            "Saving fetched papers to Zotero under collection path: %s",
            self.collection_path,
        )
        return zotero.Zotero(self.cfg.user_id, self.cfg.library_type, self.cfg.api_key)

    def _validate_papers(self) -> None:
        """Validate that papers exist to save."""
        if not self.fetched_papers:
            raise ValueError(
                "No fetched papers were found to save. "
                "Please retrieve papers using Zotero Read or Semantic Scholar first."
            )

    def _find_collection(self) -> str:
        """Find or create the target collection."""
        matched_collection_key = find_or_create_collection(
            self.zot, self.normalized_path, create_missing=False
        )

        if not matched_collection_key:
            available_collections = self.zot.collections()
            collection_names = [col["data"]["name"] for col in available_collections]
            names_display = ", ".join(collection_names)

            raise ValueError(
                f"Error: The collection path '{self.collection_path}' does "
                f"not exist in Zotero. "
                f"Available collections are: {names_display}. "
                f"Please try saving to one of these existing collections."
            )

        return matched_collection_key

    def _format_papers_for_zotero(self, matched_collection_key: str) -> None:
        """Format papers for Zotero and assign to the specified collection."""
        for paper_id, paper in self.fetched_papers.items():
            title = paper.get("Title", "N/A")
            abstract = paper.get("Abstract", "N/A")
            publication_date = paper.get("Publication Date", "N/A")
            url = paper.get("URL", "N/A")
            citations = paper.get("Citation Count", "N/A")
            venue = paper.get("Venue", "N/A")
            publication_venue = paper.get("Publication Venue", "N/A")
            journal_name = paper.get("Journal Name", "N/A")
            journal_volume = paper.get("Journal Volume", "N/A")
            journal_pages = paper.get("Journal Pages", "N/A")

            authors = [
                (
                    {
                        "creatorType": "author",
                        "firstName": name.split(" ")[0],
                        "lastName": " ".join(name.split(" ")[1:]),
                    }
                    if " " in name
                    else {"creatorType": "author", "lastName": name}
                )
                for name in [
                    author.split(" (ID: ")[0] for author in paper.get("Authors", [])
                ]
            ]

            self.zotero_items.append(
                {
                    "itemType": "journalArticle",
                    "title": title,
                    "abstractNote": abstract,
                    "date": publication_date,
                    "url": url,
                    "extra": f"Paper ID: {paper_id}\nCitations: {citations}",
                    "collections": [matched_collection_key],
                    "publicationTitle": (
                        publication_venue if publication_venue != "N/A" else venue
                    ),
                    "journalAbbreviation": journal_name,
                    "volume": journal_volume if journal_volume != "N/A" else None,
                    "pages": journal_pages if journal_pages != "N/A" else None,
                    "creators": authors,
                }
            )

    def _save_to_zotero(self) -> None:
        """Save items to Zotero."""
        try:
            response = self.zot.create_items(self.zotero_items)
            logger.info("Papers successfully saved to Zotero: %s", response)
        except Exception as e:
            logger.error("Error saving to Zotero: %s", str(e))
            raise RuntimeError(f"Error saving papers to Zotero: {str(e)}") from e

    def _create_content(self, collection_name: str) -> None:
        """Create the content message for the response."""
        self.content = (
            f"Save was successful. Papers have been saved to Zotero collection "
            f"'{collection_name}' with the requested path '{self.get_collection_path()}'.\n"
        )
        self.content += "Summary of saved papers:\n"
        self.content += f"Number of articles saved: {self.get_paper_count()}\n"
        self.content += f"Query: {self.state.get('query', 'N/A')}\n"
        top_papers = list(self.fetched_papers.values())[:2]
        top_papers_info = "\n".join(
            [
                f"{i+1}. {paper.get('Title', 'N/A')} ({paper.get('URL', 'N/A')})"
                for i, paper in enumerate(top_papers)
            ]
        )
        self.content += "Here are a few of these articles:\n" + top_papers_info

    def process_write(self) -> Dict[str, Any]:
        """Process the write operation and return results."""
        self._validate_papers()
        matched_collection_key = self._find_collection()
        self._format_papers_for_zotero(matched_collection_key)
        self._save_to_zotero()

        # Get collection name for feedback
        collections = self.zot.collections()
        collection_name = ""
        for col in collections:
            if col["key"] == matched_collection_key:
                collection_name = col["data"]["name"]
                break

        self._create_content(collection_name)

        return {
            "content": self.content,
            "fetched_papers": self.fetched_papers,
        }

    def get_paper_count(self) -> int:
        """Get the number of papers to be saved.

        Returns:
            int: The number of papers in the fetched papers dictionary.
        """
        return len(self.fetched_papers)

    def get_collection_path(self) -> str:
        """Get the normalized collection path.

        Returns:
            str: The normalized collection path where papers will be saved.
        """
        return self.collection_path
