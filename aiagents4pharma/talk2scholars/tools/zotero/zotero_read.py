#!/usr/bin/env python3

"""
This tool is used to search for papers in Zotero library.
"""

import logging
from typing import Annotated, Any
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.read_helper import ZoteroSearchData


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
def zotero_read(
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
    # Create search data object to organize variables
    search_data = ZoteroSearchData(query, only_articles, limit, tool_call_id)

    # Process the search
    search_data.process_search()
    results = search_data.get_search_results()

    return Command(
        update={
            "article_data": results["article_data"],
            "last_displayed_papers": "article_data",
            "messages": [
                ToolMessage(
                    content=results["content"],
                    tool_call_id=tool_call_id,
                    artifact=results["article_data"],
                )
            ],
        }
    )
