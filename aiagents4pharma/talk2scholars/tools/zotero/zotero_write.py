#!/usr/bin/env python3

"""
This tool is used to save fetched papers to Zotero library after human approval.
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
from .utils.zotero_path import (
    find_or_create_collection,
    fetch_papers_for_save,
)


# pylint: disable=R0914,R0911,R0912,R0915

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSaveInput(BaseModel):
    """Input schema for the Zotero save tool."""

    tool_call_id: Annotated[str, InjectedToolCallId]
    collection_path: str = Field(
        description="The path where the paper should be saved in the Zotero library."
    )
    state: Annotated[dict, InjectedState]


@tool(args_schema=ZoteroSaveInput, parse_docstring=True)
def zotero_write(
    tool_call_id: Annotated[str, InjectedToolCallId],
    collection_path: str,
    state: Annotated[dict, InjectedState],
) -> Command[Any]:
    """
    Use this tool to save previously fetched papers from Semantic Scholar
    to a specified Zotero collection after human approval.

    This tool checks if the user has approved the save operation via the
    zotero_review. If approved, it will save the papers to the
    approved collection path.

    Args:
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        collection_path (str): The Zotero collection path where papers should be saved.
        state (Annotated[dict, InjectedState]): The state containing previously fetched papers.
        user_confirmation (str, optional): User confirmation message when interrupt is
        not available.

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

    # Use our utility function to fetch papers from state
    fetched_papers = fetch_papers_for_save(state)

    if not fetched_papers:
        raise ValueError(
            "No fetched papers were found to save. "
            "Please retrieve papers using Zotero Read or Semantic Scholar first."
        )

    # Normalize the requested collection path
    normalized_path = collection_path.rstrip("/").lower()

    # Use our utility function to find or optionally create the collection
    # First try to find the exact collection
    matched_collection_key = find_or_create_collection(
        zot, normalized_path, create_missing=False  # First try without creating
    )

    if not matched_collection_key:
        # Get all collection names without hierarchy for clearer display
        available_collections = zot.collections()
        collection_names = [col["data"]["name"] for col in available_collections]
        names_display = ", ".join(collection_names)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Error: The collection path '{collection_path}' does "
                            f"not exist in Zotero. "
                            f"Available collections are: {names_display}. "
                            f"Please try saving to one of these existing collections."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # Format papers for Zotero and assign to the specified collection
    zotero_items = []
    for paper_id, paper in fetched_papers.items():
        title = paper.get("Title", "N/A")
        abstract = paper.get("Abstract", "N/A")
        publication_date = paper.get("Publication Date", "N/A")  # Use Publication Date
        url = paper.get("URL", "N/A")
        citations = paper.get("Citation Count", "N/A")
        venue = paper.get("Venue", "N/A")
        publication_venue = paper.get("Publication Venue", "N/A")
        journal_name = paper.get("Journal Name", "N/A")
        journal_volume = paper.get("Journal Volume", "N/A")
        journal_pages = paper.get("Journal Pages", "N/A")

        # Convert Authors list to Zotero format
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

        zotero_items.append(
            {
                "itemType": "journalArticle",
                "title": title,
                "abstractNote": abstract,
                "date": publication_date,  # Now saving full publication date
                "url": url,
                "extra": f"Paper ID: {paper_id}\nCitations: {citations}",
                "collections": [matched_collection_key],
                "publicationTitle": (
                    publication_venue if publication_venue != "N/A" else venue
                ),  # Use publication venue if available
                "journalAbbreviation": journal_name,  # Save Journal Name
                "volume": (
                    journal_volume if journal_volume != "N/A" else None
                ),  # Save Journal Volume
                "pages": (
                    journal_pages if journal_pages != "N/A" else None
                ),  # Save Journal Pages
                "creators": authors,  # Save authors list properly
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
    collections = zot.collections()
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
    content += "Here are a few of these articles:\n" + top_papers_info

    # Clear the approval info so it's not reused
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=fetched_papers,
                )
            ],
            "zotero_write_approval_status": {},  # Clear approval info
        }
    )
