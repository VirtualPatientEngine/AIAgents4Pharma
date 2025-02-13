#!/usr/bin/env python3

"""
Tool for asking questions to the article.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryArticleInput(BaseModel):
    """
    Input schema for the query_articles tool.
    """
    question: Annotated[str, Field(description="User question to search articles.")]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class QueryArticle(BaseTool):
    """
    Tool to ask questions to the article.
    """
    name: str = "query_article"
    description: str = "Ask questions to the article."
    args_schema: Type[BaseModel] = QueryArticleInput

    def _run(self,
             question: str,
             state: Annotated[dict, InjectedState]):
        """
        Run the tool.

        Args:
            query (str): The search query.
        """
        logger.log(logging.INFO, "loading the article from %s", state['pdf_file_name'])
        logger.log(logging.INFO, "searching the article with the question: %s", question)
        # Load the article
        loader = PyPDFLoader(state['pdf_file_name'])
        # Load the pages of the article
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        # Set up text embedding model
        text_embedding_model = state['text_embedding_model']
        logging.info("Loaded text embedding model %s", text_embedding_model)
        # Create a vector store from the pages
        vector_store = InMemoryVectorStore.from_documents(
                                            pages,
                                            text_embedding_model)
        # Search the article with the question
        docs = vector_store.similarity_search(question)
        # Return the content of the pages
        return "\n".join([doc.page_content for doc in docs])
