#!/usr/bin/env python3

"""
multi_paper_rec: Tool for getting recommendations
                based on multiple papers
"""

import json
import logging
from typing import Annotated, Any, Dict, List, Optional
import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiPaperRecInput(BaseModel):
    """Input schema for multiple paper recommendations tool."""

    paper_ids: List[str] = Field(
        description=("List of Semantic Scholar Paper IDs to get recommendations for")
    )
    limit: int = Field(
        default=2,
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


# Load hydra configuration
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(
        config_name="config", overrides=["tools/multi_paper_recommendation=default"]
    )
    cfg = cfg.tools.multi_paper_recommendation


@tool(args_schema=MultiPaperRecInput)
def get_multi_paper_recommendations(
    paper_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
    year: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get paper recommendations based on multiple papers.

    Args:
        paper_ids (List[str]): The list of paper IDs to base recommendations on.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of recommendations to return. Defaults to 2.
        year (str, optional): Year range for papers.
        Supports formats like "2024-", "-2024", "2024:2025". Defaults to None.

    Returns:
        Dict[str, Any]: The recommendations and related information.
    """
    logging.info("Starting multi-paper recommendations search.")

    endpoint = cfg.api_endpoint
    headers = cfg.headers
    payload = {"positivePaperIds": paper_ids, "negativePaperIds": []}
    params = {
        "limit": min(limit, 500),
        "fields": ",".join(cfg.api_fields),
    }

    # Add year parameter if provided
    if year:
        params["year"] = year

    # Getting recommendations
    response = requests.post(
        endpoint,
        headers=headers,
        params=params,
        data=json.dumps(payload),
        timeout=cfg.request_timeout,
    )
    logging.info(
        "API Response Status for multi-paper recommendations: %s", response.status_code
    )

    data = response.json()
    recommendations = data.get("recommendedPapers", [])

    # Create a dictionary to store the papers
    filtered_papers = {
        paper["paperId"]: {
            "Title": paper.get("title", "N/A"),
            "Abstract": paper.get("abstract", "N/A"),
            "Year": paper.get("year", "N/A"),
            "Citation Count": paper.get("citationCount", "N/A"),
            "URL": paper.get("url", "N/A"),
        }
        for paper in recommendations
        if paper.get("title") and paper.get("paperId")
    }

    return Command(
        update={
            "multi_papers": filtered_papers,  # Now sending the dictionary directly
            "messages": [
                ToolMessage(
                    content=f"Search Successful: {filtered_papers}",
                    tool_call_id=tool_call_id
                )
            ],
        }
    )
