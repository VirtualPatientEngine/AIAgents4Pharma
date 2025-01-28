"""
Embedding class using Ollama model based on LangChain Embeddings class.
"""

import time
from typing import List
import subprocess
import ollama
from langchain_ollama import OllamaEmbeddings
from .embeddings import Embeddings

class EmbeddingWithOllama(Embeddings):
    """
    Embedding class using Ollama model based on LangChain Embeddings class.
    """
    def __init__(self, model_name: str):
        """
        Initialize the EmbeddingWithOllama class.

        Args:
            model_name: The name of the Ollama model to be used.
        """
        # Setup the Ollama server
        self.__setup(model_name)

        # Set parameters
        self.model_name = model_name

        # Prepare model
        self.model = OllamaEmbeddings(model=self.model_name)

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

    def embed_documents(self, texts: List[str]) -> List[float]:
        """
        Generate embedding for a list of input texts using Ollama model.

        Args:
            texts: The list of texts to be embedded.

        Returns:
            The list of embeddings for the given texts.
        """

        # Generate the embedding
        embeddings = self.model.embed_documents(texts)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for an input text using Ollama model.

        Args:
            text: A query to be embedded.
        Returns:
            The embeddings for the given query.
        """

        # Generate the embedding
        embeddings = self.model.embed_query(text)

        return embeddings
