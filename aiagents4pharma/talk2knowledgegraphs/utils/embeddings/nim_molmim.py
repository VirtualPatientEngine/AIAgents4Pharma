"""
Embedding class using MOLMIM model from NVIDIA NIM.
"""

import json
from typing import List
import requests
from .embeddings import Embeddings

class EmbeddingWithMOLMIM(Embeddings):
    """
    Embedding class using MOLMIM model from NVIDIA NIM
    """
    def __init__(self, base_url: str):
        """
        Initialize the EmbeddingWithMOLMIM class.

        Args:
            base_url: The base URL for the NIM/MOLMIM model.
        """
        # Set base URL
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[float]:
        """
        Generate embedding for a list of SMILES strings using MOLMIM model.

        Args:
            texts: The list of SMILES strings to be embedded.

        Returns:
            The list of embeddings for the given SMILES strings.
        """
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = json.dumps({"sequences": texts})
        response = requests.post(self.base_url, headers=headers, data=data, timeout=60)
        embeddings = response.json()["embeddings"]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for an input query using MOLMIM model.

        Args:
            text: A query to be embedded.
        Returns:
            The embeddings for the given query.
        """
        # Generate the embedding
        embeddings = self.embed_documents([text])
        return embeddings
