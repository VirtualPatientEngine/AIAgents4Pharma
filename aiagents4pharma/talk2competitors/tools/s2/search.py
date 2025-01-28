#!/usr/bin/env python3

"""
This tool is used to search for academic papers on Semantic Scholar.
"""

import logging
from typing import Annotated, Any, Dict, Optional

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Input schema for the search papers tool."""

    query: str = Field(
        description="Search query string to find academic papers."
        "Be specific and include relevant academic terms."
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    year: Optional[str] = Field(
        default=None,
        description="Year range in format: YYYY for specific year, "
        "YYYY- for papers after year, -YYYY for papers before year, or YYYY:YYYY for range",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(args_schema=SearchInput)
def search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
    year: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for academic papers on Semantic Scholar.

    Args:
        query (str): The search query string to find academic papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 2.
        year (str, optional): Year range for papers.
        Supports formats like "2024-", "-2024", "2024:2025". Defaults to None.

    Returns:
        Dict[str, Any]: The search results and related information.
    """
    print("Starting paper search...")
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        # "fields": "paperId,title,abstract,year,authors,
        # citationCount,url,publicationTypes,openAccessPdf",
        "fields": "paperId,title,abstract,year,authors,citationCount,url",
    }

    # Add year parameter if provided
    if year:
        params["year"] = year

    response = requests.get(endpoint, params=params, timeout=10)
    data = response.json()
    papers = data.get("data", [])

    # Create a dictionary to store the papers
    filtered_papers = {
        paper["paperId"]: {
            "Title": paper.get("title", "N/A"),
            "Abstract": paper.get("abstract", "N/A"),
            "Year": paper.get("year", "N/A"),
            "Citation Count": paper.get("citationCount", "N/A"),
            "URL": paper.get("url", "N/A"),
            # "Publication Type": paper.get("publicationTypes", ["N/A"])[0]
            # if paper.get("publicationTypes")
            # else "N/A",
            # "Open Access PDF": paper.get("openAccessPdf", {}).get("url", "N/A")
            # if paper.get("openAccessPdf") is not None
            # else "N/A",
        }
        for paper in papers
        if paper.get("title") and paper.get("authors")
    }

    df = pd.DataFrame(filtered_papers)

    # Format papers for state update
    papers = [
        f"Paper ID: {paper_id}\n"
        f"Title: {paper_data['Title']}\n"
        f"Abstract: {paper_data['Abstract']}\n"
        f"Year: {paper_data['Year']}\n"
        f"Citations: {paper_data['Citation Count']}\n"
        f"URL: {paper_data['URL']}\n"
        # f"Publication Type: {paper_data['Publication Type']}\n"
        # f"Open Access PDF: {paper_data['Open Access PDF']}"
        for paper_id, paper_data in filtered_papers.items()
    ]

    markdown_table = df.to_markdown(tablefmt="grid")
    logging.info("Search results: %s", papers)

    return Command(
        update={
            "papers": filtered_papers,  # Now sending the dictionary directly
            "messages": [
                ToolMessage(content=markdown_table, tool_call_id=tool_call_id)
            ],
        }
    )
