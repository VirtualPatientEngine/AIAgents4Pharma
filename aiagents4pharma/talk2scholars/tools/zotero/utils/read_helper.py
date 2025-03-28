#!/usr/bin/env python3

"""
Utility for zotero read tool.
"""

import logging
from typing import Any, Dict, List
import hydra
from pyzotero import zotero
from .zotero_path import get_item_collections


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSearchData:
    """Helper class to organize Zotero search-related data."""

    def __init__(
        self,
        query: str,
        only_articles: bool,
        limit: int,
        tool_call_id: str,
    ):
        self.query = query
        self.only_articles = only_articles
        self.limit = limit
        self.tool_call_id = tool_call_id
        self.cfg = self._load_config()
        self.zot = self._init_zotero_client()
        self.item_to_collections = get_item_collections(self.zot)
        self.filtered_papers = {}
        self.content = ""

    def process_search(self) -> None:
        """Process the search request and prepare results."""
        items = self._fetch_items()
        self._filter_and_format_papers(items)
        self._create_content()

    def get_search_results(self) -> Dict[str, Any]:
        """Get the search results and content."""
        return {
            "filtered_papers": self.filtered_papers,
            "content": self.content,
        }

    def _load_config(self) -> Any:
        """Load hydra configuration."""
        with hydra.initialize(version_base=None, config_path="../../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/zotero_read=default"]
            )
            logger.info("Loaded configuration for Zotero search tool")
            return cfg.tools.zotero_read

    def _init_zotero_client(self) -> zotero.Zotero:
        """Initialize Zotero client."""
        logger.info(
            "Searching Zotero for query: '%s' (only_articles: %s, limit: %d)",
            self.query,
            self.only_articles,
            self.limit,
        )
        return zotero.Zotero(self.cfg.user_id, self.cfg.library_type, self.cfg.api_key)

    def _fetch_items(self) -> List[Dict[str, Any]]:
        """Fetch items from Zotero."""
        try:
            if self.query.strip() == "":
                logger.info(
                    "Empty query provided, fetching all items up to max_limit: %d",
                    self.cfg.zotero.max_limit,
                )
                items = self.zot.items(limit=self.cfg.zotero.max_limit)
            else:
                items = self.zot.items(
                    q=self.query, limit=min(self.limit, self.cfg.zotero.max_limit)
                )
        except Exception as e:
            logger.error("Failed to fetch items from Zotero: %s", e)
            raise RuntimeError(
                "Failed to fetch items from Zotero. Please retry the same query."
            ) from e

        logger.info("Received %d items from Zotero", len(items))

        if not items:
            logger.error("No items returned from Zotero for query: '%s'", self.query)
            raise RuntimeError(
                "No items returned from Zotero. Please retry the same query."
            )

        return items

    def _filter_and_format_papers(self, items: List[Dict[str, Any]]) -> None:
        """Filter and format papers from items."""
        filter_item_types = (
            self.cfg.zotero.filter_item_types if self.only_articles else []
        )
        logger.debug("Filtering item types: %s", filter_item_types)

        for item in items:
            if not isinstance(item, dict):
                continue

            data = item.get("data")
            if not isinstance(data, dict):
                continue

            item_type = data.get("itemType", "N/A")
            logger.debug("Item type: %s", item_type)

            key = data.get("key")
            if not key:
                continue

            collection_paths = self.item_to_collections.get(key, ["/Unknown"])

            self.filtered_papers[key] = {
                "Title": data.get("title", "N/A"),
                "Abstract": data.get("abstractNote", "N/A"),
                "Publication Date": data.get("date", "N/A"),
                "URL": data.get("url", "N/A"),
                "Type": item_type if isinstance(item_type, str) else "N/A",
                "Collections": collection_paths,
                "Citation Count": data.get("citationCount", "N/A"),
                "Venue": data.get("venue", "N/A"),
                "Publication Venue": data.get("publicationTitle", "N/A"),
                "Journal Name": data.get("journalAbbreviation", "N/A"),
                "Authors": [
                    f"{creator.get('firstName', '')} {creator.get('lastName', '')}".strip()
                    for creator in data.get("creators", [])
                    if isinstance(creator, dict)
                    and creator.get("creatorType") == "author"
                ],
            }

        if not self.filtered_papers:
            logger.error(
                "No matching papers returned from Zotero for query: '%s'", self.query
            )
            raise RuntimeError(
                "No matching papers returned from Zotero. Please retry the same query."
            )

        logger.info("Filtered %d items", len(self.filtered_papers))

    def _create_content(self) -> None:
        """Create the content message for the response."""
        top_papers = list(self.filtered_papers.values())[:2]
        top_papers_info = "\n".join(
            [
                f"{i+1}. {paper['Title']} ({paper['Type']})"
                for i, paper in enumerate(top_papers)
            ]
        )

        self.content = "Retrieval was successful. Papers are attached as an artifact."
        self.content += " And here is a summary of the retrieval results:\n"
        self.content += f"Number of papers found: {len(self.filtered_papers)}\n"
        self.content += f"Query: {self.query}\n"
        self.content += "Here are a few of these papers:\n" + top_papers_info
