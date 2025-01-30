"""
Test cases for utils/embeddings/huggingface.py
"""

import pytest
from ..utils.embeddings.huggingface import EmbeddingWithHuggingFace

@pytest.fixture(name="embedding_model")
def embedding_model_fixture():
    """Return the configuration object for the HuggingFace embedding model and model object"""
    return EmbeddingWithHuggingFace(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_cache_dir="../../cache",
        truncation=True,
    )

def test_embedding_with_huggingface_embed_documents(embedding_model):
    """Test embedding documents using the EmbeddingWithHuggingFace class."""
    # Perform embedding
    texts = ["Adalimumab", "Infliximab", "Vedolizumab"]
    result = embedding_model.embed_documents(texts)
    # Check the result
    assert len(result) == 3
    assert len(result[0]) == 768

def test_embedding_with_huggingface_embed_query(embedding_model):
    """Test embedding a query using the EmbeddingWithHuggingFace class."""
    # Perform embedding
    text = "Adalimumab"
    result = embedding_model.embed_query(text)
    # Check the result
    assert len(result) == 768

def test_embedding_with_huggingface_failed():
    """Test embedding documents using the EmbeddingWithHuggingFace class."""
    # Check if the model is available on HuggingFace Hub
    model_name = "aiagents4pharma/embeddings"
    err_msg = f"Model {model_name} is not available on HuggingFace Hub."
    with pytest.raises(ValueError, match=err_msg):
        EmbeddingWithHuggingFace(
            model_name=model_name,
            model_cache_dir="../../cache",
            truncation=True,
        )
