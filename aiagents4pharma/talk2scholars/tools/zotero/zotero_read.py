#!/usr/bin/env python3

"""
This tool is used to search for papers in Zotero library.
"""

import logging
from typing import Annotated, Any
import hydra
from pyzotero import zotero
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path import (
    get_item_collections,
)

# pylint: disable=R0914,R0912,R0915

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSearchInput(BaseModel):
    """Input schema for the Zotero search tool."""

    query: str = Field(
        description="Search query string to find papers in Zotero library."
    )
    only_articles: bool = Field(
        default=True,
        description="Whether to only search for journal articles/conference papers.",
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(args_schema=ZoteroSearchInput, parse_docstring=True)
def zotero_search_tool(
    query: str,
    only_articles: bool,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Command[Any]:
    """
    Use this tool to search and retrieve papers from Zotero library.

    Args:
        query (str): The search query string to find papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 2.

    Returns:
        Dict[str, Any]: The search results and related information.
    """
    # Load hydra configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/zotero_read=default"]
        )
        logger.info("Loaded configuration for Zotero search tool")
        cfg = cfg.tools.zotero_read
        logger.info(
            "Searching Zotero for query: '%s' (only_articles: %s, limit: %d)",
            query,
            only_articles,
            limit,
        )

    # Initialize Zotero client
    zot = zotero.Zotero(cfg.user_id, cfg.library_type, cfg.api_key)

    # Fetch collection mapping once
    item_to_collections = get_item_collections(zot)

    # If the query is empty, fetch all items (up to max_limit), otherwise use the query
    try:
        if query.strip() == "":
            logger.info(
                "Empty query provided, fetching all items up to max_limit: %d",
                cfg.zotero.max_limit,
            )
            items = zot.items(limit=cfg.zotero.max_limit)
        else:
            items = zot.items(q=query, limit=min(limit, cfg.zotero.max_limit))
    except Exception as e:
        logger.error("Failed to fetch items from Zotero: %s", e)
        raise RuntimeError(
            "Failed to fetch items from Zotero. Please retry the same query."
        ) from e

    logger.info("Received %d items from Zotero", len(items))

    if not items:
        logger.error("No items returned from Zotero for query: '%s'", query)
        raise RuntimeError(
            "No items returned from Zotero. Please retry the same query."
        )

    # Define filter criteria
    filter_item_types = cfg.zotero.filter_item_types if only_articles else []
    filter_excluded_types = (
        cfg.zotero.filter_excluded_types
    )  # Exclude non-research items

    # Filter and format papers
    filtered_papers = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        data = item.get("data")
        if not isinstance(data, dict):
            continue

        item_type = data.get("itemType")
        logger.debug("Item type: %s", item_type)

        # Exclude attachments, notes, and other unwanted types
        if (
            not item_type
            or not isinstance(item_type, str)
            or item_type in filter_excluded_types  # Skip attachments & notes
            or (
                only_articles and item_type not in filter_item_types
            )  # Skip non-research types
        ):
            continue

        key = data.get("key")
        if not key:
            continue

        # Use the imported utility function's mapping to get collection paths
        collection_paths = item_to_collections.get(key, ["/Unknown"])

        filtered_papers[key] = {
            "Title": data.get("title", "N/A"),
            "Abstract": data.get("abstractNote", "N/A"),
            "Date": data.get("date", "N/A"),
            "URL": data.get("url", "N/A"),
            "Type": item_type if isinstance(item_type, str) else "N/A",
            "Collections": collection_paths,  # Now displays full paths
        }

    if not filtered_papers:
        logger.error("No matching papers returned from Zotero for query: '%s'", query)
        raise RuntimeError(
            "No matching papers returned from Zotero. Please retry the same query."
        )

    logger.info("Filtered %d items", len(filtered_papers))

    # Prepare content with top 2 paper titles and types
    top_papers = list(filtered_papers.values())[:2]
    top_papers_info = "\n".join(
        [
            f"{i+1}. {paper['Title']} ({paper['Type']})"
            for i, paper in enumerate(top_papers)
        ]
    )

    content = "Retrieval was successful. Papers are attached as an artifact."
    content += " And here is a summary of the retrieval results:\n"
    content += f"Number of papers found: {len(filtered_papers)}\n"
    content += f"Query: {query}\n"
    content += "Top papers:\n" + top_papers_info

    return Command(
        update={
            "zotero_read": filtered_papers,
            "last_displayed_papers": "zotero_read",
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=filtered_papers,
                )
            ],
        }
    )
