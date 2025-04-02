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

# pylint: disable=broad-exception-caught


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
        self.article_data = {}
        self.content = ""

    def process_search(self) -> None:
        """Process the search request and prepare results."""
        items = self._fetch_items()
        self._filter_and_format_papers(items)
        self._create_content()

    def get_search_results(self) -> Dict[str, Any]:
        """Get the search results and content."""
        return {
            "article_data": self.article_data,
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

            self.article_data[key] = {
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
                "source": "zotero",  # Adding source field with value "zotero"
            }

            # Find PDF attachment URLs for this item
            self._find_pdf_urls(key)

        if not self.article_data:
            logger.error(
                "No matching papers returned from Zotero for query: '%s'", self.query
            )
            raise RuntimeError(
                "No matching papers returned from Zotero. Please retry the same query."
            )

        logger.info("Filtered %d items", len(self.article_data))

    def _find_pdf_urls(self, item_key: str) -> None:
        """
        Find PDF attachment URL for a specific item and add it directly to the article data.
        Only the first PDF attachment's information will be used.

        Args:
            item_key (str): The Zotero item key to find PDF URLs for
        """
        logger.info("Finding PDF URL for item: %s", item_key)

        try:
            # Get all child items (attachments) for this item
            try:
                children = self.zot.children(item_key)
            except Exception as e:
                # If we can't get children, the item might not support children
                logger.debug("Cannot get children for item %s: %s", item_key, str(e))
                return

            # Filter for PDF attachments
            pdf_attachments = [
                child
                for child in children
                if (
                    isinstance(child, dict)
                    and child.get("data", {}).get("contentType") == "application/pdf"
                )
            ]

            if not pdf_attachments:
                logger.info("No PDF attachments found for item: %s", item_key)
                return

            # Use only the first PDF attachment
            if pdf_attachments:
                attachment = pdf_attachments[0]
                attachment_data = attachment.get("data", {})
                url = attachment_data.get("url", "")

                if url:
                    # Add PDF information directly to the article entry
                    self.article_data[item_key]["filename"] = attachment_data.get(
                        "filename", "unknown.pdf"
                    )
                    self.article_data[item_key]["pdf_url"] = url
                    self.article_data[item_key]["attachment_key"] = attachment_data.get(
                        "key", ""
                    )

                    logger.info("Found PDF URL for item: %s", item_key)

        except Exception as e:
            logger.error("Error finding PDF URLs for item %s: %s", item_key, e)

    def _create_content(self) -> None:
        """Create the content message for the response."""
        top_papers = list(self.article_data.values())[:2]
        top_papers_info = "\n".join(
            [
                f"{i+1}. {paper['Title']} ({paper['Type']})"
                for i, paper in enumerate(top_papers)
            ]
        )

        self.content = "Retrieval was successful. Papers are attached as an artifact."
        self.content += " And here is a summary of the retrieval results:\n"
        self.content += f"Number of papers found: {len(self.article_data)}\n"
        self.content += f"Query: {self.query}\n"
        self.content += "Here are a few of these papers:\n" + top_papers_info
