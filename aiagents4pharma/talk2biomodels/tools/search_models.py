#!/usr/bin/env python3

"""
Tool for searching models based on search query.
"""

from typing import Type, Annotated
import logging
from pydantic import BaseModel, Field
from basico import biomodels
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchModelsInput(BaseModel):
    """
    Input schema for the search models tool.
    """
    query: str = Field(description="Search models query", default=None)
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SearchModelsTool(BaseTool):
    """
    Tool for returning the search results based on the search query.
    """
    name: str = "search_models"
    description: str = "Search models in the BioMmodels database based on keywords."
    args_schema: Type[BaseModel] = SearchModelsInput
    return_direct: bool = False

    def _run(self,
             query: str,
             state: Annotated[dict, InjectedState]) -> dict:
        """
        Run the tool.

        Args:
            query (str): The search query.

        Returns:
            dict: The answer to the question in the form of a dictionary.
        """
        logger.log(logging.INFO, "Searching models with the query and model: %s, %s",
                   query, state['llm_model'])
        search_results = biomodels.search_for_model(query)
        llm = state['llm_model']
        # Check if run_manager's metadata has the key 'prompt_content'
        prompt_content = f'''
                        Convert the input into a table.

                        The table must contain the following columns:
                        - #
                        - BioModel ID
                        - BioModel Name
                        - Format
                        - Submission Date

                        Additional Guidelines:
                        - The column # must contain the row number starting from 1.
                        - Embed the url for each BioModel ID in the table 
                        in the first column in the markdown format.
                        - The Submission Date must be in the format YYYY-MM-DD.

                        Input:
                        {input}.
                        '''
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt_content),
             ("user", "{input}")]
        )
        parser = StrOutputParser()
        chain = prompt_template | llm | parser
        return chain.invoke({"input": search_results})
