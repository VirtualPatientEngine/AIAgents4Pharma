#!/usr/bin/env python3

'''
This tool is used to return recommendations for a single paper.
'''

import logging
from typing import Annotated, Any, Dict
import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SinglePaperRecInput(BaseModel):
    """Input schema for single paper recommendation tool."""

    paper_id: str = Field(
        description="Semantic Scholar Paper ID to get recommendations for (40-character string)"
    )
    limit: int = Field(
        default=2,
        description="Maximum number of recommendations to return",
        ge=1,
        le=500,
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    model_config = {"arbitrary_types_allowed": True}


@tool(args_schema=SinglePaperRecInput)
def get_single_paper_recommendations(
                    paper_id: str,
                    tool_call_id: Annotated[str, InjectedToolCallId],
                    limit: int = 2,
                ) -> Dict[str, Any]:
    """
    Get paper recommendations based on a single paper.

    Args:
        paper_id (str): The Semantic Scholar Paper ID to get recommendations for.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of recommendations to return. Defaults to 2.

    Returns:
        Dict[str, Any]: The recommendations and related information.
    """
    logger.info("Starting single paper recommendations search.")

    endpoint = (
        f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
    )
    params = {
        "limit": min(limit, 500),  # Max 500 per API docs
        "fields": "title,paperId,abstract,year",
        "from": "all-cs",  # Using all-cs pool as specified in docs
    }

    response = requests.get(endpoint, params=params, timeout=10)
    # print(f"API Response Status: {response.status_code}")
    logging.info("API Response Status for recommendations of paper %s: %s",
                 paper_id,
                 response.status_code)
    # print(f"Request params: {params}")
    logging.info("Request params: %s", params)

    data = response.json()
    recommendations = data.get("recommendedPapers", [])

    # Extract paper ID and title from recommendations
    filtered_papers = [
        {"Paper ID": paper["paperId"], "Title": paper["title"]}
        for paper in recommendations
        if paper.get("title") and paper.get("paperId")
    ]

    # Create a DataFrame for pretty printing
    df = pd.DataFrame(filtered_papers)

    papers = [
        f"Paper ID: {paper['Paper ID']}\nTitle: {paper['Title']}"
        for paper in filtered_papers
    ]

    # Convert DataFrame to markdown table
    markdown_table = df.to_markdown(tablefmt="grid")

    return Command(
        update={
            "papers": papers,
            "messages": [
                ToolMessage(
                    content=markdown_table, tool_call_id=tool_call_id
                )
            ],
        }
    )
