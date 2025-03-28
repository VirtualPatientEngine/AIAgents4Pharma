#!/usr/bin/env python3

"""
This tool is used to save fetched papers to Zotero library after human approval.
"""

import logging
from typing import Annotated, Any
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from .utils.write_helper import ZoteroWriteData


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
    # Create write data object to organize variables
    write_data = ZoteroWriteData(tool_call_id, collection_path, state)

    try:
        # Process the write operation
        results = write_data.process_write()

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=results["content"],
                        tool_call_id=tool_call_id,
                        artifact=results["fetched_papers"],
                    )
                ],
                "zotero_write_approval_status": {},  # Clear approval info
            }
        )
    except ValueError as e:
        # Only handle collection not found errors with a Command
        if "collection path" in str(e).lower():
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=str(e),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        # Let other ValueErrors (like no papers) propagate up
        raise
