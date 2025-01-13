#!/usr/bin/env python3

"""
Tool for searching models based on search query.
"""

from urllib.error import URLError
from time import sleep
from typing import Type
from pydantic import BaseModel, Field
from basico import biomodels
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class SearchModelsInput(BaseModel):
    """
    Input schema for the search models tool.
    """
    query: str = Field(description="Search models query", default=None)

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SearchModelsTool(BaseTool):
    """
    Tool for returning the search results based on the search query.
    """
    name: str = "search_models"
    description: str = "Search models based on search query."
    args_schema: Type[BaseModel] = SearchModelsInput
    return_direct: bool = True

    def _run(self, query: str) -> str:
        """
        Run the tool.

        Args:
            query (str): The search query.

        Returns:
            str: The answer to the question.
        """
        attempts = 0
        max_retries = 3
        while attempts < max_retries:
            try:
                search_results = biomodels.search_for_model(query)
                break
            except URLError as e:
                attempts += 1
                sleep(10)
                if attempts >= max_retries:
                    raise e
        llm = ChatOpenAI(model="gpt-4o-mini")
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

    def get_metadata(self):
        """
        Get metadata for the tool.

        Returns:
            dict: The metadata for the tool.
        """
        return {
            "name": self.name,
            "description": self.description
        }
