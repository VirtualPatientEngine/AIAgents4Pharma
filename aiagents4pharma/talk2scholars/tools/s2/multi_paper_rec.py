#!/usr/bin/env python3

"""
Recommend research papers related to a set of input papers using Semantic Scholar.

Given a list of Semantic Scholar paper IDs, this tool aggregates related works
(citations and references) from each input paper and returns a consolidated list
of recommended papers.
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
    """Defines the input schema for the multi-paper recommendation tool.

    Attributes:
        paper_ids: List of 40-character Semantic Scholar Paper IDs (provide at least two).
        limit: Maximum total number of recommendations to return (1-500).
        year: Optional publication year filter; supports formats:
            'YYYY', 'YYYY-', '-YYYY', 'YYYY:YYYY'.
        tool_call_id: Internal tool call identifier injected by the system.
    """

    paper_ids: List[str] = Field(
        description="List of 40-character Semantic Scholar Paper IDs"
        "(at least two) to base recommendations on"
    )
    limit: int = Field(
        default=10,
        description="Maximum total number of recommendations to return (1-500)",
        ge=1,
        le=500,
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year filter; supports formats:"
        "'YYYY', 'YYYY-', '-YYYY', 'YYYY:YYYY'",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    model_config = {"arbitrary_types_allowed": True}


@tool(
    args_schema=MultiPaperRecInput,
    parse_docstring=True,
)
def get_multi_paper_recommendations(
    paper_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 10,
    year: Optional[str] = None,
) -> Command[Any]:
    """
    Return recommended papers based on multiple Semantic Scholar paper IDs.

    This tool accepts a list of Semantic Scholar paper IDs and returns a set of
    recommended papers by aggregating related works (citations and references)
    from each input paper.

    Args:
        paper_ids (List[str]): List of 40-character Semantic Scholar paper IDs.
        Provide at least two IDs.
        tool_call_id (str): Internal tool call identifier injected by the system.
        limit (int, optional): Maximum total number of recommendations to return. Defaults to 10.
        year (str, optional): Publication year filter; supports formats: 'YYYY',
        'YYYY-', '-YYYY', 'YYYY:YYYY'. Defaults to None.

    Returns:
        Command: A Command object containing:
            - multi_papers: List of recommended papers.
            - last_displayed_papers: Same list for display purposes.
            - messages: List containing a ToolMessage with recommendations details.
    """
    # Create recommendation data object to organize variables
    rec_data = MultiPaperRecData(paper_ids, limit, year, tool_call_id)

    # Process the recommendations
    results = rec_data.process_recommendations()

    return Command(
        update={
            "multi_papers": results["papers"],
            # Store the latest multi-paper results mapping directly for display
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
