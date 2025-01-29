#!/usr/bin/env python3

"""
Enrichment class using Ollama model based on LangChain Enrichment class.
"""

import time
from typing import List
import subprocess
import ast
import ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .enrichments import Enrichments

class EnrichmentWithOllama(Enrichments):
    """
    Enrichment class using Ollama model based on the Enrichment abstract class.
    """
    def __init__(
        self,
        model_name: str,
        prompt_enrichment: str,
        temperature: float,
        streaming: bool,
    ):
        """
        Initialize the EnrichmentWithOllama class.

        Args:
            model_name: The name of the Ollama model to be used.
            prompt_enrichment: The prompt enrichment template.
            temperature: The temperature for the Ollama model.
            streaming: The streaming flag for the Ollama model.
        """
        # Setup the Ollama server
        self.__setup(model_name)

        # Set parameters
        self.model_name = model_name
        self.prompt_enrichment = prompt_enrichment
        self.temperature = temperature
        self.streaming = streaming

        # Prepare prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_enrichment),
                ("human", "{input}"),
            ]
        )

        # Prepare model
        self.model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming,
        )

    def __setup(self, model_name: str) -> None:
        """
        Check if the Ollama model is available and run the Ollama server if needed.

        Args:
            model_name: The name of the Ollama model to be used.
        """
        try:
            models_list = ollama.list()["models"]
            if model_name not in [m['model'].replace(":latest", "") for m in models_list]:
                ollama.pull(model_name)
                time.sleep(30)
                raise ValueError(f"Pulled {model_name} model")
        except Exception as e:
            with subprocess.Popen(
                "ollama serve", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ):
                time.sleep(10)
            raise ValueError(f"Error: {e} and restarted Ollama server.") from e

    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input texts with additional textual features using OLLAMA model.
        Important: Make sure the input is a list of texts based on the defined prompt template
        with 'input' as the variable name.

        Args:
            texts: The list of texts to be enriched.

        Returns:
            The list of enriched texts.
        """

        # Perform enrichment
        chain = self.prompt_template | self.model | StrOutputParser()

        # Generate the enriched node
        # Important: Make sure the input is a list of texts based on the defined prompt template
        # with 'input' as the variable name
        enriched_texts = chain.invoke({"input": "[" + ", ".join(texts) + "]"})

        # Convert the enriched nodes to a list of dictionary
        enriched_texts = ast.literal_eval(enriched_texts.replace("```", ""))

        # Final check for the enriched texts
        assert len(enriched_texts) == len(texts)

        return enriched_texts

    def enrich_documents_with_rag(self, texts, docs):
        """
        Enrich a list of input texts with additional textual features using OLLAMA model with RAG.
        As of now, we don't have a RAG model to test this method yet.
        Thus, we will just call the enrich_documents method instead.

        Args:
            texts: The list of texts to be enriched.
            docs: The list of reference documents to enrich the input texts.
        
        Returns:
            The list of enriched texts
        """
        return self.enrich_documents(texts)
