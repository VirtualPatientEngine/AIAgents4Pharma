import json
import time
from typing import Annotated, Any, Dict, List

import pandas as pd
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator


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

    @classmethod
    @field_validator("paper_ids")
    def validate_paper_ids(cls, v: List[str]) -> List[str]:
        """
        Validates the list of paper IDs.

        Args:
            v (List[str]): The list of paper IDs to validate.

        Returns:
            List[str]: The validated list of paper IDs.

        Raises:
            ValueError: If the list is empty, contains more than 10 IDs, or any ID has an invalid format.
        """
        if not v:
            raise ValueError("At least one paper ID must be provided")
        if len(v) > 10:
            raise ValueError("Maximum of 10 paper IDs allowed")
        return v

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
    # Validate inputs
    if not paper_ids:
        raise ValueError("At least one paper ID must be provided")
    if len(paper_ids) > 10:
        raise ValueError("Maximum of 10 paper IDs allowed")
    print("Starting multi-paper recommendations search...")

    endpoint = "https://api.semanticscholar.org/recommendations/v1/papers"
    headers = {"Content-Type": "application/json"}
    payload = {"positivePaperIds": paper_ids, "negativePaperIds": []}
    params = {"limit": min(limit, 500), "fields": "title,paperId"}

    max_retries = 3
    retry_count = 0
    retry_delay = 2

    while retry_count < max_retries:
        print(f"Attempt {retry_count + 1} of {max_retries}")
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                params=params,
                data=json.dumps(payload),
                timeout=10,
            )
            print(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendedPapers", [])

                if not recommendations:
                    print("No recommendations found")
                    return Command(
                        update={
                            "papers": [
                                "No recommendations found for the provided papers"
                            ],
                            "messages": [
                                ToolMessage(
                                    content="No recommendations found for the provided papers",
                                    tool_call_id=tool_call_id,
                                )
                            ],
                        }
                    )

                # Create a list to store the papers
                papers_list = []
                for paper in recommendations:
                    if paper.get("title") and paper.get("paperId"):
                        papers_list.append(
                            {"Paper ID": paper["paperId"], "Title": paper["title"]}
                        )

                if not papers_list:
                    return Command(
                        update={
                            "papers": ["No valid recommendations found"],
                            "messages": [
                                ToolMessage(
                                    content="No valid recommendations found",
                                    tool_call_id=tool_call_id,
                                )
                            ],
                        }
                    )

                df = pd.DataFrame(papers_list)
                print("Created DataFrame with results:")
                print(df)

                # Format papers for state update
                formatted_papers = [
                    f"Paper ID: {paper['Paper ID']}\nTitle: {paper['Title']}"
                    for paper in papers_list
                ]

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

            if response.status_code == 404:
                return Command(
                    update={
                        "papers": ["One or more paper IDs not found"],
                        "messages": [
                            ToolMessage(
                                content="One or more paper IDs not found",
                                tool_call_id=tool_call_id,
                            )
                        ],
                    }
                )

            retry_count += 1
            if retry_count < max_retries:
                wait_time = retry_delay * (2**retry_count)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        except Exception as e:
            print(f"Error: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                return Command(
                    update={
                        "papers": [f"Error getting recommendations: {str(e)}"],
                        "messages": [
                            ToolMessage(
                                content=f"Error getting recommendations: {str(e)}",
                                tool_call_id=tool_call_id,
                            )
                        ],
                    }
                )
            time.sleep(retry_delay * (2**retry_count))

    return Command(
        update={
            "papers": ["Failed to get recommendations after maximum retries"],
            "messages": [
                ToolMessage(
                    content="Failed to get recommendations after maximum retries",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
