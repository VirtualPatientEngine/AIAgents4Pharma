#!/usr/bin/env python3

"""
Resolve a paper title to a Semantic Scholar paperId.

This module provides a tool that queries the Semantic Scholar API for the best match to a
given paper title (full or partial) and returns the corresponding `paperId` string.
Configuration is loaded via Hydra and the top ranked result is returned.
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
    Input schema for title→paperId resolution.

    Fields
    -------
    paper_title : str
        Paper title to search. Accepts full titles or informative partial titles.
    tool_call_id : InjectedToolCallId
        Runtime-injected identifier for tracing the tool invocation.
    """

    paper_title: str = Field(..., description="The paper title to search for on Semantic Scholar.")
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
    Look up a Semantic Scholar paperId from a paper title.

    Behavior
    --------
    - Loads Hydra config from `tools.retrieve_semantic_scholar_paper_id`.
    - Sends a search request with `query=<paper_title>`, `limit=1`, and requested fields.
    - Parses the top hit and returns its `paperId` as the ToolMessage content (plain string).

    Parameters
    ----------
    paper_title : str
        Title or informative partial title to resolve.
    tool_call_id : str
        Runtime-injected identifier for the tool call.

    Returns
    -------
    Command
        update = {
          "messages": [
            ToolMessage(
              content="<paperId>",  # Semantic Scholar paperId string
              tool_call_id=<tool_call_id>
            )
          ]
        }

    Exceptions
    ----------
    ValueError
        Raised when no match is found for the provided title.
    requests.RequestException
        Raised on network/HTTP errors (timeout, connection issues, etc.).

    Examples
    --------
    >>> retrieve_semantic_scholar_paper_id("Attention Is All You Need", "tc_123")
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
