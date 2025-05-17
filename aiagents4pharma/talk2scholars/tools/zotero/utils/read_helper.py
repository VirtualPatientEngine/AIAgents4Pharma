#!/usr/bin/env python3

"""
Utility for zotero read tool.
"""

import logging
from typing import Any, Dict, List

import hydra
import requests
from pyzotero import zotero

from .zotero_path import get_item_collections
from .zotero_pdf_downloader import download_pdfs_in_parallel

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
        download_pdfs: bool = True,
        **_kwargs,
    ):
        self.query = query
        self.only_articles = only_articles
        self.limit = limit
        # Control whether to fetch PDF attachments now
        self.download_pdfs = download_pdfs
        self.cfg = self._load_config()
        self.zot = self._init_zotero_client()
        self.item_to_collections = get_item_collections(self.zot)
        self.article_data = {}
        self.content = ""
        # Create a session for connection pooling
        self.session = requests.Session()

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

    def _collect_item_attachments(self) -> Dict[str, str]:
        """Collect PDF attachment keys for non-orphan items."""
        item_attachments: Dict[str, str] = {}
        for item_key, item_data in self.article_data.items():
            if item_data.get("Type") == "orphan_attachment":
                continue
            try:
                children = self.zot.children(item_key)
                for child in children:
                    data = child.get("data", {})
                    if data.get("contentType") == "application/pdf":
                        attachment_key = data.get("key")
                        filename = data.get("filename", "unknown.pdf")
                        if attachment_key:
                            item_attachments[attachment_key] = item_key
                            self.article_data[item_key]["filename"] = filename
                            break
            except Exception as e:
                logger.error("Failed to get attachments for item %s: %s", item_key, e)
        return item_attachments

    def _process_orphaned_pdfs(self, orphaned_pdfs: Dict[str, str]) -> None:
        """Download or record orphaned PDF attachments."""
        if self.download_pdfs:
            logger.info("Downloading %d orphaned PDFs in parallel", len(orphaned_pdfs))
            results = download_pdfs_in_parallel(
                self.session,
                self.cfg.user_id,
                self.cfg.api_key,
                orphaned_pdfs,
                chunk_size=getattr(self.cfg, "chunk_size", None),
            )
            for item_key, (file_path, filename, attachment_key) in results.items():
                self.article_data[item_key]["filename"] = filename
                self.article_data[item_key]["pdf_url"] = file_path
                self.article_data[item_key]["attachment_key"] = attachment_key
                logger.info("Downloaded orphaned Zotero PDF to: %s", file_path)
        else:
            logger.info("Skipping orphaned PDF downloads (download_pdfs=False)")
            for attachment_key in orphaned_pdfs:
                self.article_data[attachment_key]["attachment_key"] = attachment_key
                self.article_data[attachment_key]["filename"] = (
                    self.article_data[attachment_key].get("Title", attachment_key)
                )

    def _process_item_pdfs(self, item_attachments: Dict[str, str]) -> None:
        """Download or record regular item PDF attachments."""
        if self.download_pdfs:
            logger.info(
                "Downloading %d regular item PDFs in parallel", len(item_attachments)
            )
            results = download_pdfs_in_parallel(
                self.session,
                self.cfg.user_id,
                self.cfg.api_key,
                item_attachments,
                chunk_size=getattr(self.cfg, "chunk_size", None),
            )
        else:
            logger.info("Skipping regular PDF downloads (download_pdfs=False)")
            results = {}
            for attachment_key, item_key in item_attachments.items():
                self.article_data[item_key]["attachment_key"] = attachment_key
        for item_key, (file_path, filename, attachment_key) in results.items():
            self.article_data[item_key]["filename"] = filename
            self.article_data[item_key]["pdf_url"] = file_path
            self.article_data[item_key]["attachment_key"] = attachment_key
            logger.info("Downloaded Zotero PDF to: %s", file_path)

    def _filter_and_format_papers(self, items: List[Dict[str, Any]]) -> None:
        """Filter and format papers from Zotero items, including standalone PDFs."""
        filter_item_types = (
            self.cfg.zotero.filter_item_types if self.only_articles else []
        )
        logger.debug("Filtering item types: %s", filter_item_types)

        # Maps to track attachments for batch processing
        orphaned_pdfs: Dict[str, str] = {}  # attachment_key -> item key (same for orphans)

        # First pass: process all items without downloading PDFs
        for item in items:
            if not isinstance(item, dict):
                continue

            data = item.get("data", {})
            item_type = data.get("itemType", "N/A")
            key = data.get("key")
            if not key:
                continue

            # CASE 1: Top-level item (e.g., journalArticle)
            if item_type != "attachment":
                collection_paths = self.item_to_collections.get(key, ["/Unknown"])

                self.article_data[key] = {
                    "Title": data.get("title", "N/A"),
                    "Abstract": data.get("abstractNote", "N/A"),
                    "Publication Date": data.get("date", "N/A"),
                    "URL": data.get("url", "N/A"),
                    "Type": item_type,
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
                    "source": "zotero",
                }
                # We'll collect attachment info in second pass

            # CASE 2: Standalone orphaned PDF attachment
            elif data.get("contentType") == "application/pdf" and not data.get(
                "parentItem"
            ):
                attachment_key = key
                filename = data.get("filename", "unknown.pdf")

                # Add to orphaned PDFs for batch processing
                orphaned_pdfs[attachment_key] = (
                    attachment_key  # Same key as both attachment and "item"
                )

                # Create the entry without PDF info yet
                self.article_data[key] = {
                    "Title": filename,
                    "Abstract": "No abstract available",
                    "Publication Date": "N/A",
                    "URL": "N/A",
                    "Type": "orphan_attachment",
                    "Collections": ["/(No Collection)"],
                    "Citation Count": "N/A",
                    "Venue": "N/A",
                    "Publication Venue": "N/A",
                    "Journal Name": "N/A",
                    "Authors": ["(Unknown)"],
                    "source": "zotero",
                }

        # Collect and process attachments
        item_attachments = self._collect_item_attachments()

        # Process orphaned PDFs
        self._process_orphaned_pdfs(orphaned_pdfs)

        # Process regular item PDFs
        self._process_item_pdfs(item_attachments)

        # Ensure we have some results
        if not self.article_data:
            logger.error(
                "No matching papers returned from Zotero for query: '%s'", self.query
            )
            raise RuntimeError(
                "No matching papers returned from Zotero. Please retry the same query."
            )

        logger.info(
            "Filtered %d items (including orphaned attachments)", len(self.article_data)
        )

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
