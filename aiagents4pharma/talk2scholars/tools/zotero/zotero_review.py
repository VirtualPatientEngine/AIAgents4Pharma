#!/usr/bin/env python3

"""
This tool implements human-in-the-loop review for Zotero write operations.
"""

import logging
from typing import Annotated, Any, Optional, Literal
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from .utils.zotero_path import fetch_papers_for_save

# pylint: disable=R0914,R0911,R0912,R0915
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroReviewDecision(BaseModel):
    """
    Structured output schema for the human review decision.
    - decision: "approve", "reject", or "custom"
    - custom_path: Optional custom collection path if the decision is "custom"
    """

    decision: Literal["approve", "reject", "custom"]
    custom_path: Optional[str] = None


class ZoteroReviewInput(BaseModel):
    """Input schema for the Zotero review tool."""

    tool_call_id: Annotated[str, InjectedToolCallId]
    collection_path: str = Field(
        description="The path where the paper should be saved in the Zotero library."
    )
    state: Annotated[dict, InjectedState]


@tool(args_schema=ZoteroReviewInput, parse_docstring=True)
def zotero_review(
    tool_call_id: Annotated[str, InjectedToolCallId],
    collection_path: str,
    state: Annotated[dict, InjectedState],
) -> Command[Any]:
    """
    Use this tool to get human review and approval before saving papers to Zotero.
    This tool should be called before the zotero_write to ensure the user approves
    the operation.

    Args:
        tool_call_id (str): The tool call ID.
        collection_path (str): The Zotero collection path where papers should be saved.
        state (dict): The state containing previously fetched papers.

    Returns:
        Command[Any]: The next action to take based on human input.
    """
    logger.info("Requesting human review for saving to collection: %s", collection_path)

    # Use our utility function to fetch papers from state
    fetched_papers = fetch_papers_for_save(state)

    if not fetched_papers:
        raise ValueError(
            "No fetched papers were found to save. "
            "Please retrieve papers using Zotero Read or Semantic Scholar first."
        )

    # Prepare papers summary for review
    papers_summary = []
    for paper_id, paper in list(fetched_papers.items())[
        :5
    ]:  # Limit to 5 papers for readability
        logger.info("Paper ID: %s", paper_id)
        title = paper.get("Title", "N/A")
        authors = ", ".join(
            [author.split(" (ID: ")[0] for author in paper.get("Authors", [])[:2]]
        )
        if len(paper.get("Authors", [])) > 2:
            authors += " et al."
        papers_summary.append(f"- {title} by {authors}")

    if len(fetched_papers) > 5:
        papers_summary.append(f"... and {len(fetched_papers) - 5} more papers")

    papers_preview = "\n".join(papers_summary)
    total_papers = len(fetched_papers)

    # Prepare review information
    review_info = {
        "action": "save_to_zotero",
        "collection_path": collection_path,
        "total_papers": total_papers,
        "papers_preview": papers_preview,
        "message": (
            f"Would you like to save {total_papers} papers to Zotero "
            f"collection '{collection_path}'? Please respond with a structured decision "
            f"using one of the following options: 'approve', 'reject', or 'custom' "
            f"(with a custom_path)."
        ),
    }

    try:
        # Interrupt the graph to get human approval
        human_review = interrupt(review_info)
        # Process human response using structured output via LLM
        llm_model = state.get("llm_model")
        if llm_model is None:
            raise ValueError("LLM model is not available in the state.")
        structured_llm = llm_model.with_structured_output(ZoteroReviewDecision)
        # Convert the raw human response to a message for structured parsing
        decision_response = structured_llm.invoke(
            [HumanMessage(content=str(human_review))]
        )

        # Process the structured response
        if decision_response.decision == "approve":
            logger.info("User approved saving papers to Zotero")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Human approved saving {total_papers} papers to Zotero "
                                f"collection '{collection_path}'."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "zotero_write_approval_status": {
                        "collection_path": collection_path,
                        "approved": True,
                    },
                }
            )
        if decision_response.decision == "custom" and decision_response.custom_path:
            logger.info(
                "User approved with custom path: %s", decision_response.custom_path
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Human approved saving papers to custom Zotero "
                                f"collection '{decision_response.custom_path}'."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "zotero_write_approval_status": {
                        "collection_path": decision_response.custom_path,
                        "approved": True,
                    },
                }
            )
        logger.info("User rejected saving papers to Zotero")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Human rejected saving papers to Zotero.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "zotero_write_approval_status": {"approved": False},
            }
        )
        # pylint: disable=broad-except
    except Exception as e:
        # If interrupt or structured output processing fails, fallback to explicit confirmation
        logger.warning("Structured review processing failed: %s", e)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"REVIEW REQUIRED: Would you like to save {total_papers} papers "
                            f"to Zotero collection '{collection_path}'?\n\n"
                            f"Papers to save:\n{papers_preview}\n\n"
                            "Please respond with 'Yes' to confirm or 'No' to cancel."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
                "zotero_write_approval_status": {
                    "collection_path": collection_path,
                    "papers_reviewed": True,
                    "approved": False,  # Not approved yet
                    "papers_count": total_papers,
                },
            }
        )
