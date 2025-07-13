#!/usr/bin/env python3

"""
Tool for retrieving a Semantic Scholar paper ID given a paper title.

This tool queries the Semantic Scholar API for the best match of the provided paper title
and returns the unique Semantic Scholar paperId. Use when you have a known title and need its
Semantic Scholar identifier for further metadata retrieval or pipeline integration. Do not
use this tool for broad literature search; use the `search` tool instead.
"""

import logging
from typing import Annotated, Any
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


class RetrieveSemanticScholarPaperIdInput(BaseModel):
    """
    Pydantic schema for retrieving a Semantic Scholar paper ID.

    Fields:
      paper_title: The title (full or partial) of the paper to look up on Semantic Scholar.
      tool_call_id: LangGraph-injected identifier for tracking the tool invocation.
    """

    paper_title: str = Field(
        ..., description="The paper title to search for on Semantic Scholar."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "retrieve_semantic_scholar_paper_id",
    args_schema=RetrieveSemanticScholarPaperIdInput,
    parse_docstring=True,
)
def retrieve_semantic_scholar_paper_id(
    paper_title: str,
    tool_call_id: str,
) -> Command[Any]:
    """
    Search for a paper by title on Semantic Scholar and return its unique paper ID.

    This tool issues a GET request to the Semantic Scholar API to find the best match
    for the given paper title, then returns the paper's Semantic Scholar ID.

    Use when you have a known title (full or partial) and need the Semantic Scholar ID
    to fetch additional metadata or perform downstream lookups. Do not use this tool
    for broad literature searches; for general search use the `search` tool.

    Args:
        paper_title (str): The title of the paper to look up.
        tool_call_id (str): LangGraph-injected identifier for this tool call.

    Returns:
        Command: A structured response containing a ToolMessage whose content is
          the Semantic Scholar paper ID string (e.g., 'abc123xyz').

    Raises:
        ValueError: If no matching paper is found for the given title.
        requests.RequestException: If the API request fails.
    """
    # Load hydra configuration
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tools/retrieve_semantic_scholar_paper_id=default"],
        )
        cfg = cfg.tools.retrieve_semantic_scholar_paper_id
        logger.info("Loaded configuration for Semantic Scholar paper ID retrieval tool")
    logger.info("Retrieving ID of paper with title: %s", paper_title)
    endpoint = cfg.api_endpoint
    params = {
        "query": paper_title,
        "limit": 1,
        "fields": ",".join(cfg.api_fields),
    }

    response = requests.get(endpoint, params=params, timeout=10)
    data = response.json()
    papers = data.get("data", [])
    logger.info("Received %d papers", len(papers))
    if not papers:
        logger.error("No papers found for query: %s", paper_title)
        raise ValueError(f"No papers found for query: {paper_title}. Try again.")
    # Extract the paper ID from the top result
    paper_id = papers[0]["paperId"]
    logger.info("Found paper ID: %s", paper_id)
    # Prepare the response content (just the ID)
    response_text = paper_id
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response_text,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
