#!/usr/bin/env python3

"""
Zotero Read Tool

This LangGraph tool searches a user's Zotero library for items matching a query
and optionally downloads their PDF attachments. It returns structured metadata
for each found item and makes the results available as an artifact.
"""

import logging
from typing import Annotated, Any
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.read_helper import ZoteroSearchData


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSearchInput(BaseModel):
    """Input schema for the Zotero search tool.

    Attributes:
        query (str): Search string to match against item metadata.
        only_articles (bool): If True, restrict results to 'journalArticle' and similar types.
        limit (int): Maximum number of items to fetch from Zotero.
        download_pdfs (bool): If True, download PDF attachments for each item.
        tool_call_id (str): Internal identifier for this tool invocation.
    """

    query: str = Field(
        description="Search query string to find papers in Zotero library."
    )
    only_articles: bool = Field(
        default=True,
        description="Whether to only search for journal articles/conference papers.",
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    download_pdfs: bool = Field(
        default=False,
        description="Whether to download PDF attachments immediately (default True).",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(args_schema=ZoteroSearchInput, parse_docstring=True)
def zotero_read(
    query: str,
    only_articles: bool,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
    download_pdfs: bool = False,
) -> Command[Any]:
    """
    Execute a search on the Zotero library and return matching items.

    Args:
        query (str): Text query to search in titles, abstracts, tags, etc.
        only_articles (bool): When True, only include items of type 'journalArticle'
        or 'conferencePaper'.
        tool_call_id (str): Internal ID injected by LangGraph to track this tool call.
        limit (int, optional): Max number of items to return (1–100). Defaults to 2.
        download_pdfs (bool, optional): If True, PDFs for each returned item will be downloaded now.
                                        If False, only metadata is fetched. Defaults to False.

    Returns:
        Command[Any]: A LangGraph Command updating the agent state:
            - 'article_data': dict mapping item keys to metadata (and 'pdf_url' if downloaded).
            - 'last_displayed_papers': identifier pointing to the articles in state.
            - 'messages': list containing a ToolMessage with a human-readable summary
                           and an 'artifact' referencing the raw article_data.
    """
    # Create search data object to organize variables
    # download_pdfs flag controls whether PDFs are fetched now or deferred
    search_data = ZoteroSearchData(
        query=query,
        only_articles=only_articles,
        limit=limit,
        download_pdfs=download_pdfs,
        tool_call_id=tool_call_id,
    )

    # Process the search
    search_data.process_search()
    results = search_data.get_search_results()

    return Command(
        update={
            "article_data": results["article_data"],
            # Store the latest article_data mapping directly for display
            "last_displayed_papers": results["article_data"],
            "messages": [
                ToolMessage(
                    content=results["content"],
                    tool_call_id=tool_call_id,
                    artifact=results["article_data"],
                )
            ],
        }
    )
