#!/usr/bin/env python3

"""
This tool is used to save fetched papers to Zotero library.
"""

import logging
from typing import Annotated, Any
import hydra
from pyzotero import zotero
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path import (
    get_item_collections,
)

# pylint: disable=R0914,R0912,R0915

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSaveInput(BaseModel):
    """Input schema for the Zotero save tool."""

    tool_call_id: Annotated[str, InjectedToolCallId]
    collection_path: str = Field(
        default=None,
        description=(
            "The path where the paper should be saved in the Zotero library."
            "Example: '/machine/cern/mobile'"
        ),
    )
    state: Annotated[dict, InjectedState]


@tool(args_schema=ZoteroSaveInput, parse_docstring=True)
def zotero_save_tool(
    tool_call_id: Annotated[str, InjectedToolCallId],
    collection_path: str,
    state: Annotated[dict, InjectedState],
) -> Command[Any]:
    """
    Use this tool to save previously fetched papers from Semantic Scholar
    to a specified Zotero collection.

    Args:
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        collection_path (str): The Zotero collection path where papers should be saved.
        state (Annotated[dict, InjectedState]): The state containing previously fetched papers.

    Returns:
        Command[Any]: The save results and related information.
    """
    # Load hydra configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/zotero_write=default"]
        )
        cfg = cfg.tools.zotero_write
        logger.info("Loaded configuration for Zotero write tool")
    logger.info(
        "Saving fetched papers to Zotero under collection path: %s", collection_path
    )

    # Initialize Zotero client
    zot = zotero.Zotero(cfg.user_id, cfg.library_type, cfg.api_key)

    # Retrieve last displayed papers from the agent state
    last_displayed_key = state.get("last_displayed_papers", {})
    if isinstance(last_displayed_key, str):
        # If it's a string (key to another state object), get that object
        fetched_papers = state.get(last_displayed_key, {})
        logger.info("Using papers from '%s' state key", last_displayed_key)
    else:
        # If it's already the papers object
        fetched_papers = last_displayed_key
        logger.info("Using papers directly from last_displayed_papers")

    if not fetched_papers:
        logger.warning("No fetched papers found to save.")
        raise RuntimeError(
            "No fetched papers were found to save. Please retry the same query."
        )

    # First, check if zotero_read exists in state and has collection data
    zotero_read_data = state.get("zotero_read", {})
    logger.info("Retrieved zotero_read from state: %d items", len(zotero_read_data))

    # If zotero_read is empty, use get_item_collections as fallback
    if not zotero_read_data:
        logger.info(
            "zotero_read is empty, fetching paths dynamically using get_item_collections"
        )
        try:
            zotero_read_data = get_item_collections(zot)
            logger.info(
                "Successfully generated %d path mappings", len(zotero_read_data)
            )
        except Exception as e:
            logger.error("Error generating path mappings: %s", str(e))
            raise RuntimeError(
                "Failed to generate collection path mappings. Please retry the same query."
            ) from e

    # Get all collections to find the correct one
    collections = zot.collections()
    logger.info("Found %d collections", len(collections))

    # Normalize the requested collection path (remove trailing slash, lowercase for comparison)
    normalized_path = collection_path.rstrip("/").lower()

    # Find matching collection
    matched_collection_key = None

    # First, try to directly find the collection key in zotero_read data
    for key, paths in zotero_read_data.items():
        if isinstance(paths, list):
            for path in paths:
                if path.lower() == normalized_path:
                    matched_collection_key = key
                    logger.info(
                        "Found direct match in zotero_read: %s -> %s", path, key
                    )
                    break
        elif isinstance(paths, str) and paths.lower() == normalized_path:
            matched_collection_key = key
            logger.info("Found direct match in zotero_read: %s -> %s", paths, key)
            break

    # If not found in zotero_read, try matching by collection name
    if not matched_collection_key:
        for col in collections:
            col_name = col["data"]["name"]
            if f"/{col_name}".lower() == normalized_path:
                matched_collection_key = col["key"]
                logger.info(
                    "Found direct match by collection name: %s (key: %s)",
                    col_name,
                    col["key"],
                )
                break

    # If still not found, try part-matching
    if not matched_collection_key:
        name_to_key = {col["data"]["name"].lower(): col["key"] for col in collections}
        collection_name = normalized_path.lstrip("/")
        if collection_name in name_to_key:
            matched_collection_key = name_to_key[collection_name]
            logger.info(
                "Found match by collection name: %s -> %s",
                collection_name,
                matched_collection_key,
            )
        else:
            path_parts = normalized_path.strip("/").split("/")
            for part in path_parts:
                if part in name_to_key:
                    matched_collection_key = name_to_key[part]
                    logger.info(
                        "Found match by path component: %s -> %s",
                        part,
                        matched_collection_key,
                    )
                    break

    # Do not fall back to a default collection: raise error if no match found
    if not matched_collection_key:
        logger.error(
            "Invalid collection path: %s. No matching collection found in Zotero.",
            collection_path,
        )

        available_paths = ", ".join(["/" + col["data"]["name"] for col in collections])
        raise RuntimeError(
            f"Error: The collection path '{collection_path}' does not exist in Zotero. "
            f"Available collections are: {available_paths}"
        )

    # Format papers for Zotero and assign to the specified collection
    zotero_items = []
    for paper_id, paper in fetched_papers.items():
        title = paper.get("Title", paper.get("title", "N/A"))
        abstract = paper.get("Abstract", paper.get("abstractNote", "N/A"))
        date = paper.get("Date", paper.get("date", "N/A"))
        url = paper.get("URL", paper.get("url", paper.get("URL", "N/A")))
        citations = paper.get("Citations", "N/A")

        zotero_items.append(
            {
                "itemType": "journalArticle",
                "title": title,
                "abstractNote": abstract,
                "date": date,
                "url": url,
                "extra": f"Paper ID: {paper_id}\nCitations: {citations}",
                "collections": [matched_collection_key],
            }
        )

    # Save items to Zotero
    try:
        response = zot.create_items(zotero_items)
        logger.info("Papers successfully saved to Zotero: %s", response)
    except Exception as e:
        logger.error("Error saving to Zotero: %s", str(e))
        raise RuntimeError(f"Error saving papers to Zotero: {str(e)}") from e

    # Get the collection name for better feedback
    collection_name = ""
    for col in collections:
        if col["key"] == matched_collection_key:
            collection_name = col["data"]["name"]
            break

    content = (
        f"Save was successful. Papers have been saved to Zotero collection '{collection_name}' "
        f"with the requested path '{collection_path}'.\n"
    )
    content += "Summary of saved papers:\n"
    content += f"Number of articles saved: {len(fetched_papers)}\n"
    content += f"Query: {state.get('query', 'N/A')}\n"
    top_papers = list(fetched_papers.values())[:2]
    top_papers_info = "\n".join(
        [
            f"{i+1}. {paper.get('Title', 'N/A')} ({paper.get('URL', 'N/A')})"
            for i, paper in enumerate(top_papers)
        ]
    )
    content += "Here are the top articles:\n" + top_papers_info

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=fetched_papers,
                )
            ],
        }
    )
