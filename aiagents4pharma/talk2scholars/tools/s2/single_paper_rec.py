#!/usr/bin/env python3

"""
Recommend research papers related to a single input paper using Semantic Scholar.

Given a Semantic Scholar paper ID, this tool retrieves related works
(citations and references) and returns a curated list of recommended papers.
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
    """Defines the input schema for the single-paper recommendation tool.

    Attributes:
        paper_id: 40-character Semantic Scholar Paper ID to base recommendations on.
        limit: Maximum number of recommendations to return (1-500).
        year: Optional publication year filter; supports 'YYYY', 'YYYY-', '-YYYY', 'YYYY:YYYY'.
        tool_call_id: Internal tool call identifier injected by the system.
    """

    paper_id: str = Field(
        description="40-character Semantic Scholar Paper ID to base recommendations on"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of recommendations to return (1-500)",
        ge=1,
        le=500,
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year filter; supports formats::"
        "'YYYY', 'YYYY-', '-YYYY', 'YYYY:YYYY'",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    model_config = {"arbitrary_types_allowed": True}


@tool(
    args_schema=SinglePaperRecInput,
    parse_docstring=True,
)
def get_single_paper_recommendations(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 10,
    year: Optional[str] = None,
) -> Command[Any]:
    """
    Return recommended papers for a single Semantic Scholar paper ID.

    This tool accepts a single Semantic Scholar paper ID and returns related works
    by aggregating citations and references.

    Args:
        paper_id (str): 40-character Semantic Scholar paper ID.
        tool_call_id (str): Internal tool call identifier injected by the system.
        limit (int, optional): Maximum number of recommendations to return. Defaults to 5.
        year (str, optional): Publication year filter; supports 'YYYY', 'YYYY-',
        '-YYYY', 'YYYY:YYYY'. Defaults to None.

    Returns:
        Command: A Command object containing:
            - papers: List of recommended papers.
            - last_displayed_papers: Same list for display purposes.
            - messages: List containing a ToolMessage with recommendation details.
    """
    # Create recommendation data object to organize variables
    rec_data = SinglePaperRecData(paper_id, limit, year, tool_call_id)

    # Process the recommendations
    results = rec_data.process_recommendations()

    return Command(
        update={
            "papers": results["papers"],
            # Store the latest single-paper results mapping directly for display
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
