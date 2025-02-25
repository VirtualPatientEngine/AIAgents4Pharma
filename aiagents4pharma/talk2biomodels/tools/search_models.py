#!/usr/bin/env python3

"""
Tool for searching models based on search query.
"""

from typing import Type, Annotated
import logging
from pydantic import BaseModel, Field
import pandas as pd
from basico import biomodels
from langgraph.types import Command
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchModelsInput(BaseModel):
    """
    Input schema for the search models tool.
    """
    query: str = Field(description="Search models query", default=None)
    num_query: int = Field(description="Top number of models to search",
                           default=10,
                           le=100)
    tool_call_id: Annotated[str, InjectedToolCallId]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SearchModelsTool(BaseTool):
    """
    Tool for returning the search results based on the search query.
    """
    name: str = "search_models"
    description: str = "Search for only manually curated models in "
    "the BioMmodels database based on keywords."
    args_schema: Type[BaseModel] = SearchModelsInput
    return_direct: bool = False

    def _run(self,
             tool_call_id: Annotated[str, InjectedToolCallId],
             query: str = None,
             num_query: int = 10) -> dict:
        """
        Run the tool.

        Args:
            query (str): The search query.
            num_query (int): The number of models to search.
            tool_call_id (str): The tool call ID.

        Returns:
            dict: The answer to the question in the form of a dictionary.
        """
        logger.log(logging.INFO, "Searching models with the query and number %s, %s",
                   query, num_query)
        # Search for models based on the query
        search_results = biomodels.search_for_model(query, num_results=num_query)
        # Convert the search results to a pandas DataFrame
        df = pd.DataFrame(search_results)
        # Prepare a message to return
        first_n = min(3, len(search_results))
        content = f"Found {len(search_results)} manually curated models"
        content += f" for the query: {query}."
        # Pass the first 3 models to the LLM
        # to avoid hallucinations
        content += f" Here is the summary of the first {first_n} models:"
        for i in range(first_n):
            content += f"\nModel {i+1}: {search_results[i]['name']} (ID: {search_results[i]['id']})"
        # Return the updated state of the tool
        return Command(
                update={
                    # update the message history
                    "messages": [
                        ToolMessage(
                            content=content,
                            tool_call_id=tool_call_id,
                            artifact={'dic_data': df.to_dict(orient='records')}
                            )
                        ],
                    }
            )
