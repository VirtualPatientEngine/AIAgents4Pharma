#!/usr/bin/env python3

"""
Embedding class using SentenceTransformer model based on LangChain Embeddings class.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from .embeddings import Embeddings


class EmbeddingWithSentenceTransformer(Embeddings):
    """
    Embedding class using SentenceTransformer model based on LangChain Embeddings class.
    """

    def __init__(
        self,
        model_name: str,
        model_cache_dir: str = None,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the EmbeddingWithSentenceTransformer class.

        Args:
            model_name: The name of the SentenceTransformer model to be used.
            model_cache_dir: The directory to cache the SentenceTransformer model.
            trust_remote_code: Whether to trust the remote code of the model.
        """

        # Set parameters
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.trust_remote_code = trust_remote_code

        # Load the model
        self.model = SentenceTransformer(self.model_name,
                                         cache_folder=self.model_cache_dir,
                                         trust_remote_code=self.trust_remote_code)

    def embed_documents(self, texts: List[str]) -> List[float]:
        """
        Generate embedding for a list of input texts using SentenceTransformer model.

        Args:
            texts: The list of texts to be embedded.

        Returns:
            The list of embeddings for the given texts.
        """

        # Generate the embedding
        embeddings = self.model.encode(texts, show_progress_bar=False)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for an input text using SentenceTransformer model.

        Args:
            text: A query to be embedded.
        Returns:
            The embeddings for the given query.
        """

        # Generate the embedding
        embeddings = self.model.encode(text, show_progress_bar=False)

        return embeddings
