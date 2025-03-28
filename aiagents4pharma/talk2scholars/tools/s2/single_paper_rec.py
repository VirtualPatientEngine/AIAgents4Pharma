#!/usr/bin/env python3

"""
This tool is used to return recommendations for a single paper.
"""

import logging
from typing import Annotated, Any, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.single_helper import SinglePaperRecData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SinglePaperRecInput(BaseModel):
    """Input schema for single paper recommendation tool."""

    paper_id: str = Field(
        description="Semantic Scholar Paper ID to get recommendations for (40-character string)"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of recommendations to return",
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


@tool(args_schema=SinglePaperRecInput, parse_docstring=True)
def get_single_paper_recommendations(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 10,
    year: Optional[str] = None,
) -> Command[Any]:
    """
    Get recommendations for a single paper using its Semantic Scholar ID.
    No other ID types are supported.

    Args:
        paper_id (str): The Semantic Scholar Paper ID to get recommendations for.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of recommendations to return. Defaults to 5.
        year (str, optional): Year range for papers.
        Supports formats like "2024-", "-2024", "2024:2025". Defaults to None.

    Returns:
        Dict[str, Any]: The recommendations and related information.
    """
    # Create recommendation data object to organize variables
    rec_data = SinglePaperRecData(paper_id, limit, year, tool_call_id)

    # Process the recommendations
    results = rec_data.process_recommendations()

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
