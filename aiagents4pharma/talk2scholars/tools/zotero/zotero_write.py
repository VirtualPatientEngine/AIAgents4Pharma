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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSaveInput(BaseModel):
    """Input schema for the Zotero save tool."""

    tool_call_id: Annotated[str, InjectedToolCallId]


# Load hydra configuration
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(config_name="config", overrides=["tools/zotero_save=default"])
    cfg = cfg.tools.zotero_save


@tool(args_schema=ZoteroSaveInput, parse_docstring=True)
def zotero_save_tool(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command[Any]:
    """
    Use this tool to save previously fetched papers from Semantic Scholar to Zotero.

    Args:
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        state (dict): The state containing previously fetched papers.

    Returns:
        Dict[str, Any]: The save results and related information.
    """
    logger.info("Saving fetched papers to Zotero.")

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

    # Fetch collections from zotero_read state
    zotero_read = state.get("zotero_read", {})

    # Format papers for Zotero and assign to collections from zotero_read
    zotero_items = []
    for paper_id, paper in fetched_papers.items():
        collection_names = zotero_read.get(paper_id, {}).get("Collections", ["Unknown"])

        zotero_items.append(
            {
                "itemType": "journalArticle",
                "title": paper.get("Title", "N/A"),
                "abstractNote": paper.get("Abstract", "N/A"),
                "date": paper.get("Date", "N/A"),
                "url": paper.get("URL", "N/A"),
                "extra": f"Citations: {paper.get('Citations', 'N/A')}",
                "collections": collection_names,
            }
        )

    # Save items to Zotero
    response = zot.create_items(zotero_items)
    logger.info("Papers successfully saved to Zotero.")

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Successfully saved fetched papers to Zotero.",
                    tool_call_id=tool_call_id,
                    artifact=response,
                )
            ]
        }
    )
