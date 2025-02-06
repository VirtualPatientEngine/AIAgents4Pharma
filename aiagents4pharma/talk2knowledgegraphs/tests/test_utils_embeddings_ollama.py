"""
Test cases for utils/embeddings/ollama.py
"""

import pytest
import ollama
from ..utils.embeddings.ollama import EmbeddingWithOllama

@pytest.fixture(name="ollama_config")
def fixture_ollama_config():
    """Return a dictionary with Ollama configuration."""
    return {
        "model_name": "all-minilm", # Choose a small model
    }

def test_no_model_ollama(ollama_config):
    """Test the case when the Ollama model is not available."""
    cfg = ollama_config

    # Delete the Ollama model
    try:
        ollama.delete(cfg["model_name"])
    except ollama.ResponseError:
        pass

    # Check if the model is available
    with pytest.raises(
        ValueError, match=f"Error: Pulled {cfg["model_name"]} model and restarted Ollama server."
    ):
        EmbeddingWithOllama(model_name=cfg["model_name"])

@pytest.fixture(name="embedding_model")
def embedding_model_fixture(ollama_config):
    """Return the configuration object for the Ollama embedding model and model object"""
    cfg = ollama_config
    return EmbeddingWithOllama(model_name=cfg["model_name"])

def test_embedding_with_ollama_embed_documents(embedding_model):
    """Test embedding documents using the EmbeddingWithOllama class."""
    # Perform embedding
    texts = ["Adalimumab", "Infliximab", "Vedolizumab"]
    result = embedding_model.embed_documents(texts)
    # Check the result
    assert len(result) == 3
    assert len(result[0]) == 384

def test_embedding_with_ollama_embed_query(embedding_model):
    """Test embedding a query using the EmbeddingWithOllama class."""
    # Perform embedding
    text = "Adalimumab"
    result = embedding_model.embed_query(text)
    # Check the result
    assert len(result) == 384

    # Delete the Ollama model so that it will not be cached afterward
    ollama.delete(embedding_model.model_name)
