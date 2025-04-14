#!/usr/bin/env python3

"""
Utility for zotero read tool.
"""

import logging
import tempfile
from typing import Any, Dict, List, Tuple, Optional

import hydra
import requests
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

    def _download_zotero_pdf(self, attachment_key: str) -> Optional[Tuple[str, str]]:
        """Download a PDF from Zotero by attachment key. Returns (file_path, filename) or None."""
        zotero_pdf_url = (
            f"https://api.zotero.org/users/{self.cfg.user_id}/items/"
            f"{attachment_key}/file"
        )
        headers = {"Zotero-API-Key": self.cfg.api_key}

        try:
            response = requests.get(
                zotero_pdf_url, headers=headers, stream=True, timeout=10
            )
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

            content_disp = response.headers.get("Content-Disposition", "")
            filename = (
                content_disp.split("filename=")[-1].strip('"')
                if "filename=" in content_disp
                else "downloaded.pdf"
            )

            return temp_file_path, filename

        except Exception as e:
            logger.error(
                "Failed to download Zotero PDF for attachment %s: %s", attachment_key, e
            )
            return None

    def _filter_and_format_papers(self, items: List[Dict[str, Any]]) -> None:
        """Filter and format papers from Zotero items, including standalone PDFs."""
        filter_item_types = (
            self.cfg.zotero.filter_item_types if self.only_articles else []
        )
        logger.debug("Filtering item types: %s", filter_item_types)

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

                # Try to fetch attached PDF
                self._find_pdf_urls(key)

            # CASE 2: Standalone orphaned PDF attachment
            elif data.get("contentType") == "application/pdf" and not data.get(
                "parentItem"
            ):
                attachment_key = key
                filename = data.get("filename", "unknown.pdf")

                result = self._download_zotero_pdf(attachment_key)
                if result:
                    temp_file_path, resolved_filename = result
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
                        "source": "zotero_orphan",
                        "filename": resolved_filename,
                        "pdf_url": temp_file_path,
                        "attachment_key": attachment_key,
                    }
                    logger.info("Downloaded orphaned Zotero PDF to: %s", temp_file_path)

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

    def _find_pdf_urls(self, item_key: str) -> None:
        """Download Zotero-hosted PDF for a specific item and update article_data."""
        logger.info("Attempting to download Zotero PDF for item: %s", item_key)

        try:
            children = self.zot.children(item_key)
            pdf_attachments = [
                child
                for child in children
                if isinstance(child, dict)
                and child.get("data", {}).get("contentType") == "application/pdf"
            ]

            if not pdf_attachments:
                logger.info("No Zotero PDF attachments found for item: %s", item_key)
                return

            attachment = pdf_attachments[0]
            attachment_data = attachment.get("data", {})
            attachment_key = attachment_data.get("key")
            filename = attachment_data.get("filename", "unknown.pdf")

            if not attachment_key:
                logger.warning("No attachment key found for PDF in item: %s", item_key)
                return

            result = self._download_zotero_pdf(attachment_key)
            if result:
                temp_file_path, resolved_filename = result
                self.article_data[item_key]["filename"] = resolved_filename or filename
                self.article_data[item_key]["pdf_url"] = temp_file_path
                self.article_data[item_key]["attachment_key"] = attachment_key
                logger.info("Downloaded Zotero PDF to: %s", temp_file_path)

        except Exception as e:
            logger.error("Failed to download PDF for item %s: %s", item_key, e)

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
