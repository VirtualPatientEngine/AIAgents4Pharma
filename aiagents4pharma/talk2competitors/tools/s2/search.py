#!/usr/bin/env python3

"""
This tool is used to search for academic papers on Semantic Scholar.
"""

import logging
from typing import Annotated, Any, Dict
import pandas as pd
import requests
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from pydantic import BaseModel, Field

from ...config.config import config

class SearchInput(BaseModel):
    """Input schema for the search papers tool."""

    query: str = Field(
        description="Search query string to find academic papers."
        "Be specific and include relevant academic terms."
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool(args_schema=SearchInput)
def search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """
    Search for academic papers on Semantic Scholar.

    Args:
        query (str): The search query string to find academic papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 2.

    Returns:
        Dict[str, Any]: The search results and related information.
    """
    print("Starting paper search...")
    endpoint = f"{config.SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,title,abstract,year,authors,citationCount,openAccessPdf",
    }
    response = requests.get(endpoint, params=params, timeout=10)
    data = response.json()
    papers = data.get("data", [])

    filtered_papers = [
        {"Paper ID": paper["paperId"], "Title": paper["title"]}
        for paper in papers
        if paper.get("title") and paper.get("authors")
    ]

    df = pd.DataFrame(filtered_papers)

    papers = [
        f"Paper ID: {paper['Paper ID']}\nTitle: {paper['Title']}"
        for paper in filtered_papers
    ]

    markdown_table = df.to_markdown(tablefmt="grid")
    logging.info("Search results: %s", papers)

    return {
        "papers": papers,
        "messages": [AIMessage(content=markdown_table)],
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "search_tool",
                    "arguments": {"query": query, "limit": limit},
                },
            }
        ],
    }
