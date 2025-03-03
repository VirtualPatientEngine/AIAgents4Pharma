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


# Load hydra configuration
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(config_name="config", overrides=["tools/zotero_write=default"])
    cfg = cfg.tools.zotero_write


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
        state (dict): The state containing previously fetched papers.

    Returns:
        Dict[str, Any]: The save results and related information.
    """
    logger.info(
        f"Saving fetched papers to Zotero under collection path: {collection_path}"
    )

    # Initialize Zotero client
    zot = zotero.Zotero(cfg.user_id, cfg.library_type, cfg.api_key)

    # Retrieve last displayed papers from the agent state
    fetched_papers = state.get("last_displayed_papers", {})

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

    # Fetch all collection paths from Zotero
    item_to_collections = get_item_collections(zot)

    # Find the collection key matching the given path
    matched_collection_key = None
    for col_key, col_path in item_to_collections.items():
        if col_path == collection_path:
            matched_collection_key = col_key
            break

    # Raise error if the collection path does not exist
    if not matched_collection_key:
        logger.error(
            f"Invalid collection path: {collection_path}. No matching collection found in Zotero."
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: The collection path '{collection_path}' does not exist in Zotero.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Format papers for Zotero and assign to the specified collection
    zotero_items = []
    for paper_id, paper in fetched_papers.items():
        zotero_items.append(
            {
                "itemType": "journalArticle",
                "title": paper.get("Title", "N/A"),
                "abstractNote": paper.get("Abstract", "N/A"),
                "date": paper.get("Date", "N/A"),
                "url": paper.get("URL", "N/A"),
                "extra": f"Paper ID: {paper_id}\nCitations: {paper.get('Citations', 'N/A')}",
                "collections": [
                    matched_collection_key
                ],  # Assign to the specified collection
            }
        )

    # Save items to Zotero
    response = zot.create_items(zotero_items)
    logger.info("Papers successfully saved to Zotero.")

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Successfully saved fetched papers to Zotero under '{collection_path}'.",
                    tool_call_id=tool_call_id,
                    artifact=response,
                )
            ]
        }
    )
