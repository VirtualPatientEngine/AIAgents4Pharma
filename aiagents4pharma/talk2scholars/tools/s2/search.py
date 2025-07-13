#!/usr/bin/env python3

"""
Search for academic papers on Semantic Scholar by title or keywords.

Given a text query, this tool retrieves relevant papers from Semantic Scholar,
optionally filtered by publication year.
"""

import logging
from typing import Annotated, Any, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.search_helper import SearchData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Defines the input schema for the paper search tool.

    Attributes:
        query: Full or partial paper title or keywords to search for.
        limit: Maximum number of search results to return (1-100).
        year: Optional publication year filter; supports 'YYYY',
        'YYYY-', '-YYYY', 'YYYY:YYYY'.
        tool_call_id: Internal tool call identifier injected by the system.
    """

    query: str = Field(
        description="Full or partial paper title or keywords to search for"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of search results to return (1-100)",
        ge=1,
        le=100,
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year filter; supports formats:"
        "'YYYY', 'YYYY-', '-YYYY', 'YYYY:YYYY'",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "search_tool",
    args_schema=SearchInput,
    parse_docstring=True,
)
def search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 10,
    year: Optional[str] = None,
) -> Command[Any]:
    """
    Return academic papers from Semantic Scholar matching a title or keyword query.

    This tool searches Semantic Scholar for papers whose titles or keywords
    match the given text, optionally filtered by publication year.

    Args:
        query (str): Full or partial paper title or keywords to search for.
        tool_call_id (str): Internal tool call identifier injected by the system.
        limit (int, optional): Maximum number of search results to return. Defaults to 5.
        year (str, optional): Publication year filter; supports 'YYYY',
        'YYYY-', '-YYYY', 'YYYY:YYYY'. Defaults to None.

    Returns:
        Command: A Command object containing:
            - papers: List of matching papers.
            - last_displayed_papers: Same list for display purposes.
            - messages: List containing a ToolMessage with search results details.
    """
    # Create search data object to organize variables
    search_data = SearchData(query, limit, year, tool_call_id)

    # Process the search
    results = search_data.process_search()

    return Command(
        update={
            "papers": results["papers"],
            # Store the latest results mapping directly for display
            "last_displayed_papers": results["papers"],
            "messages": [
                ToolMessage(
                    content=results["content"],
                    tool_call_id=tool_call_id,
                    artifact=results["papers"],
                )
            ],
        }
    )
