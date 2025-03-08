#!/usr/bin/env python3

"""
This tool is used to search for academic papers on Semantic Scholar.
"""

import logging
from typing import Annotated, Any
import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import Field


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool("retrieve_semantic_scholar_paper_id", parse_docstring=True)
def retrieve_semantic_scholar_paper_id(
    tool_call_id: Annotated[str, InjectedToolCallId],
    paper_title: str = Field(
        description="The title of the paper to search for on Semantic Scholar."
    ),
) -> Command[Any]:
    """
    This tool can be used to search for a paper on Semantic Scholar
    and retrieve the paper Semantic Scholar ID.

    This is useful for when an article is retrieved from users Zotero library
    and the Semantic Scholar ID is needed to retrieve more information about the paper.

    Args:
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        paper_title (str): The title of the paper to search for on Semantic Scholar.

    Returns:
        ToolMessage: A message containing the paper ID.
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
    # Get the paper ID
    paper_id = papers[0]["paperId"]

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Paper ID for '{paper_title}' is: {paper_id}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
