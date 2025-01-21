import time
from typing import Annotated, Any, Dict

import pandas as pd
import requests
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException, tool
from langchain_core.tools.base import InjectedToolCallId
from pydantic import BaseModel, Field

from config.config import config


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

    max_retries = 3
    retry_count = 0
    retry_delay = 2
    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count + 1} of {max_retries}")
            response = requests.get(endpoint, params=params, timeout=10)
            if response.status_code == 429:
                retry_count += 1
                wait_time = retry_delay * (2**retry_count)
                print(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            if response.status_code == 200:
                print("Successful response received")
                break
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                raise ToolException(
                    f"Error searching papers after {max_retries} attempts: {str(e)}"
                ) from e
            time.sleep(retry_delay * (2**retry_count))
            continue

    print("Processing response...")
    data = response.json()
    papers = data.get("data", [])

    filtered_papers = [
        {"Paper ID": paper["paperId"], "Title": paper["title"]}
        for paper in papers
        if paper.get("title") and paper.get("authors")
    ]

    if not filtered_papers:
        return {
            "papers": ["No papers found matching your query."],
            "messages": [AIMessage(content="No papers found matching your query")],
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

    df = pd.DataFrame(filtered_papers)
    print("Created DataFrame with results")
    print(df)

    papers = [
        f"Paper ID: {paper['Paper ID']}\nTitle: {paper['Title']}"
        for paper in filtered_papers
    ]

    markdown_table = df.to_markdown(tablefmt="grid")
    print("Search tool execution completed")

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
