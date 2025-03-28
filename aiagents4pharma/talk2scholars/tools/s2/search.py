#!/usr/bin/env python3

"""
This tool is used to search for academic papers on Semantic Scholar.
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
    """Input schema for the search papers tool."""

    query: str = Field(
        description="Search query string to find academic papers."
        "Be specific and include relevant academic terms."
    )
    limit: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    year: Optional[str] = Field(
        default=None,
        description="Year range in format: YYYY for specific year, "
        "YYYY- for papers after year, -YYYY for papers before year, or YYYY:YYYY for range",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("search_tool", args_schema=SearchInput, parse_docstring=True)
def search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 5,
    year: Optional[str] = None,
) -> Command[Any]:
    """
    Search for academic papers on Semantic Scholar.

    Args:
        query (str): The search query string to find academic papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        year (str, optional): Year range for papers.
        Supports formats like "2024-", "-2024", "2024:2025". Defaults to None.

    Returns:
        The number of papers found on Semantic Scholar.
    """
    # Create search data object to organize variables
    search_data = SearchData(query, limit, year, tool_call_id)

    # Process the search
    results = search_data.process_search()

    return Command(
        update={
            "papers": results["papers"],
            "last_displayed_papers": "papers",
            "messages": [
                ToolMessage(
                    content=results["content"],
                    tool_call_id=tool_call_id,
                    artifact=results["papers"],
                )
            ],
        }
    )
