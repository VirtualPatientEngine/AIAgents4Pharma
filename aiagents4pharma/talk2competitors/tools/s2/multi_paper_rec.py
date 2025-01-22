#!/usr/bin/env python3

"""
multi_paper_rec: Tool for getting recommendations 
                based on multiple papers
"""

import logging
import json
from typing import Annotated, Any, Dict, List
import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

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
    tool_call_id: Annotated[str, InjectedToolCallId]

    model_config = {"arbitrary_types_allowed": True}


@tool(args_schema=MultiPaperRecInput)
def get_multi_paper_recommendations(
    paper_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """
    Get paper recommendations based on multiple papers.

    Args:
        paper_ids (List[str]): The list of paper IDs to base recommendations on.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of recommendations to return. Defaults to 2.

    Returns:
        Dict[str, Any]: The recommendations and related information.
    """
    logging.info("Starting multi-paper recommendations search.")

    endpoint = "https://api.semanticscholar.org/recommendations/v1/papers"
    headers = {"Content-Type": "application/json"}
    payload = {"positivePaperIds": paper_ids, "negativePaperIds": []}
    params = {"limit": min(limit, 500), "fields": "title,paperId"}

    # Getting recommendations
    response = requests.post(
        endpoint,
        headers=headers,
        params=params,
        data=json.dumps(payload),
        timeout=10,
    )
    logging.info("API Response Status for multi-paper recommendations: %s",
                 response.status_code)

    data = response.json()
    recommendations = data.get("recommendedPapers", [])

    # Create a list to store the papers
    papers_list = []
    for paper in recommendations:
        if paper.get("title") and paper.get("paperId"):
            papers_list.append(
                {"Paper ID": paper["paperId"], "Title": paper["title"]}
            )

    # Create a DataFrame from the list of papers
    df = pd.DataFrame(papers_list)
    # print("Created DataFrame with results:")
    logging.info("Created DataFrame with results: %s", df)

    # Format papers for state update
    formatted_papers = [
        f"Paper ID: {paper['Paper ID']}\nTitle: {paper['Title']}"
        for paper in papers_list
    ]

    # Convert DataFrame to markdown table
    markdown_table = df.to_markdown(tablefmt="grid")
    return Command(
        update={
            "papers": formatted_papers,
            "messages": [
                ToolMessage(
                    content=markdown_table, tool_call_id=tool_call_id
                )
            ],
        }
    )
