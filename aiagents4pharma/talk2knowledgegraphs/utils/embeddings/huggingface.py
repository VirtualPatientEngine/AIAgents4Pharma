"""
Embedding class using HuggingFace model based on LangChain Embeddings class.
"""

from typing import List
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .embeddings import Embeddings

class EmbeddingWithHuggingFace(Embeddings):
    """
    Embedding class using HuggingFace model based on LangChain Embeddings class.
    """

    def __init__(
        self,
        model_name: str,
        model_cache_dir: str = None,
        truncation: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the EmbeddingWithHuggingFace class.

        Args:
            model_name: The name of the HuggingFace model to be used.
            model_cache_dir: The directory to cache the HuggingFace model.
            truncation: The truncation flag for the HuggingFace tokenizer.
            return_tensors: The return_tensors flag for the HuggingFace tokenizer.
            device: The device to run the model on.
        """

        # Set parameters
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.truncation = truncation
        self.device = device

        # Try to load the model from HuggingFace Hub
        try:
            AutoConfig.from_pretrained(self.model_name)
        except EnvironmentError as e:
            raise ValueError(
                f"Model {self.model_name} is not available on HuggingFace Hub."
            ) from e

        # Load HuggingFace tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.model_cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir=self.model_cache_dir
        )

    def meanpooling(self, output, mask) -> torch.Tensor:
        """
        Mean Pooling - Take attention mask into account for correct averaging.
        According to the following documentation:
        https://huggingface.co/NeuML/pubmedbert-base-embeddings

        Args:
            output: The output of the model.
            mask: The mask of the model.
        """
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[float]:
        """
        Generate embedding for a list of input texts using HuggingFace model.

        Args:
            texts: The list of texts to be embedded.

        Returns:
            The list of embeddings for the given texts.
        """

        # Generate the embedding
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=self.truncation,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model.to(self.device)(**inputs)
            embeddings = self.meanpooling(outputs, inputs['attention_mask']).cpu()

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for an input text using HuggingFace model.

        Args:
            text: A query to be embedded.
        Returns:
            The embeddings for the given query.
        """

        # Generate the embedding
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=self.truncation,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model.to(self.device)(**inputs)
            embeddings = self.meanpooling(outputs, inputs['attention_mask']).cpu()[0]

        return embeddings
