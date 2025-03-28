#!/usr/bin/env python3

"""
This tool is used to return recommendations based on multiple papers
"""

import logging
from typing import Annotated, Any, List, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.multi_helper import MultiPaperRecData


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiPaperRecInput(BaseModel):
    """Input schema for multiple paper recommendations tool."""

    paper_ids: List[str] = Field(
        description="List of Semantic Scholar Paper IDs to get recommendations for"
    )
    limit: int = Field(
        default=10,
        description="Maximum total number of recommendations to return",
        ge=1,
        le=500,
    )
    year: Optional[str] = Field(
        default=None,
        description="Year range in format: YYYY for specific year, "
        "YYYY- for papers after year, -YYYY for papers before year, or YYYY:YYYY for range",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    model_config = {"arbitrary_types_allowed": True}


@tool(args_schema=MultiPaperRecInput, parse_docstring=True)
def get_multi_paper_recommendations(
    paper_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
    year: Optional[str] = None,
) -> Command[Any]:
    """
    Get recommendations for a group of multiple papers using the Semantic Scholar IDs.
    No other paper IDs are supported.

    Args:
        paper_ids (List[str]): The list of paper IDs to base recommendations on.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of recommendations to return. Defaults to 2.
        year (str, optional): Year range for papers.
        Supports formats like "2024-", "-2024", "2024:2025". Defaults to None.

    Returns:
        Dict[str, Any]: The recommendations and related information.
    """
    # Create recommendation data object to organize variables
    rec_data = MultiPaperRecData(paper_ids, limit, year, tool_call_id)

    # Process the recommendations
    results = rec_data.process_recommendations()

    return Command(
        update={
            "multi_papers": results["papers"],
            "last_displayed_papers": "multi_papers",
            "messages": [
                ToolMessage(
                    content=results["content"],
                    tool_call_id=tool_call_id,
                    artifact=results["papers"],
                )
            ],
        }
    )
