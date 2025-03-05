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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSaveInput(BaseModel):
    """Input schema for the Zotero save tool."""

    tool_call_id: Annotated[str, InjectedToolCallId]
    collection_path: str = Field(
        default="/Unknown",
        description="The path where the paper should be saved in the Zotero library. Example: '/machine/cern/mobile'.",
    )


@tool(args_schema=ZoteroSaveInput, parse_docstring=True)
def zotero_save_tool(
    tool_call_id: Annotated[str, InjectedToolCallId],
    collection_path: str,
    state: Annotated[dict, InjectedState],
) -> Command[Any]:
    """
    Use this tool to save previously fetched papers from Semantic Scholar to a specified Zotero collection.

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
        f"Saving fetched papers to Zotero under collection path: {collection_path}"
    )

    # Initialize Zotero client
    zot = zotero.Zotero(cfg.user_id, cfg.library_type, cfg.api_key)

    # Retrieve last displayed papers from the agent state
    last_displayed_key = state.get("last_displayed_papers", {})
    if isinstance(last_displayed_key, str):
        # If it's a string (key to another state object), get that object
        fetched_papers = state.get(last_displayed_key, {})
        logger.info(f"Using papers from '{last_displayed_key}' state key")
    else:
        # If it's already the papers object
        fetched_papers = last_displayed_key
        logger.info("Using papers directly from last_displayed_papers")

    if not fetched_papers:
        logger.warning("No fetched papers found to save.")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fetched papers were found to save.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # First, check if zotero_read exists in state and has collection data
    zotero_read_data = state.get("zotero_read", {})
    logger.info(f"Retrieved zotero_read from state: {len(zotero_read_data)} items")

    # If zotero_read is empty, use get_item_collections as fallback
    if not zotero_read_data:
        logger.info(
            "zotero_read is empty, fetching paths dynamically using get_item_collections"
        )
        try:
            zotero_read_data = get_item_collections(zot)
            logger.info(f"Successfully generated {len(zotero_read_data)} path mappings")
        except Exception as e:
            logger.error(f"Error generating path mappings: {str(e)}")

    # Get all collections to find the correct one
    collections = zot.collections()
    logger.info(f"Found {len(collections)} collections")

    # Normalize the requested collection path (remove trailing slash, lowercase for comparison)
    normalized_path = collection_path.rstrip("/").lower()

    # Find matching collection
    matched_collection_key = None

    # First, try to directly find the collection key in zotero_read data
    # The format might be different based on how get_item_collections works
    for key, paths in zotero_read_data.items():
        if isinstance(paths, list):
            # Handle case where values are lists of paths
            for path in paths:
                if path.lower() == normalized_path:
                    matched_collection_key = key
                    logger.info(f"Found direct match in zotero_read: {path} -> {key}")
                    break
        elif isinstance(paths, str) and paths.lower() == normalized_path:
            # Handle case where values are single paths
            matched_collection_key = key
            logger.info(f"Found direct match in zotero_read: {paths} -> {key}")

    # If not found in zotero_read, try matching by collection name
    if not matched_collection_key:
        for col in collections:
            col_name = col["data"]["name"]
            if f"/{col_name}".lower() == normalized_path:
                matched_collection_key = col["key"]
                logger.info(
                    f"Found direct match by collection name: {col_name} (key: {col['key']})"
                )
                break

    # If still not found, try part-matching
    if not matched_collection_key:
        # Create mapping from collection names to keys
        name_to_key = {col["data"]["name"].lower(): col["key"] for col in collections}

        # Check if it's a collection name (without the leading slash)
        collection_name = normalized_path.lstrip("/")
        if collection_name in name_to_key:
            matched_collection_key = name_to_key[collection_name]
            logger.info(
                f"Found match by collection name: {collection_name} -> {matched_collection_key}"
            )
        else:
            # Try to find a part of the path that matches a collection name
            path_parts = normalized_path.strip("/").split("/")
            for part in path_parts:
                if part in name_to_key:
                    matched_collection_key = name_to_key[part]
                    logger.info(
                        f"Found match by path component: {part} -> {matched_collection_key}"
                    )
                    break

    # If all else fails, use the first collection as a fallback
    if not matched_collection_key and collections:
        first_collection = collections[0]
        matched_collection_key = first_collection["key"]
        collection_name = first_collection["data"]["name"]
        logger.warning(
            f"Could not find collection '{collection_path}'. "
            f"Falling back to first collection: '{collection_name}' (key: {matched_collection_key})"
        )

    # Return error if we still can't find a collection
    if not matched_collection_key:
        logger.error(
            f"Invalid collection path: {collection_path}. No matching collection found in Zotero."
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: The collection path '{collection_path}' does not exist in Zotero. Available collections are: {', '.join(['/' + col['data']['name'] for col in collections])}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Format papers for Zotero and assign to the specified collection
    zotero_items = []
    for paper_id, paper in fetched_papers.items():
        # Support both S2 and Zotero paper formats with fallbacks
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
                "collections": [
                    matched_collection_key
                ],  # Assign to the specified collection
            }
        )

    # Save items to Zotero
    try:
        response = zot.create_items(zotero_items)
        logger.info(f"Papers successfully saved to Zotero: {response}")

        # Get the collection name for better feedback
        collection_name = ""
        for col in collections:
            if col["key"] == matched_collection_key:
                collection_name = col["data"]["name"]
                break

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Successfully saved {len(zotero_items)} papers to Zotero collection '{collection_name}'.",
                        tool_call_id=tool_call_id,
                        artifact=response,
                    )
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error saving to Zotero: {str(e)}")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error saving papers to Zotero: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
