"""
Test cases for utils/enrichments/ollama.py
"""

import pytest
import ollama
from ..utils.enrichments.ollama import EnrichmentWithOllama


@pytest.fixture(name="ollama_config")
def fixture_ollama_config():
    """Return a dictionary with Ollama configuration."""
    return {
        "model_name": "llama3.2:1b",
        "prompt_enrichment": """
            Given the input as a list of strings, please return the list of addditional information 
            of each input terms using your prior knowledge.

            Example:
            Input: ['acetaminophen', 'aspirin']
            Ouput: ['acetaminophen is a medication used to treat pain and fever',
            'aspirin is a medication used to treat pain, fever, and inflammation']

            Do not include any pretext as the output, only the list of strings enriched.

            Input: {input}
        """,
        "temperature": 0.0,
        "streaming": False,
    }


def test_no_model_ollama(ollama_config):
    """Test the case when the Ollama model is not available."""
    cfg = ollama_config
    cfg_model = "smollm2:135m"  # Choose a small model

    # Delete the Ollama model
    try:
        ollama.delete(cfg_model)
    except ollama.ResponseError:
        pass

    # Check if the model is available
    with pytest.raises(
        ValueError,
        match=f"Error: Pulled {cfg_model} model and restarted Ollama server.",
    ):
        EnrichmentWithOllama(
            model_name=cfg_model,
            prompt_enrichment=cfg["prompt_enrichment"],
            temperature=cfg["temperature"],
            streaming=cfg["streaming"],
        )
    ollama.delete(cfg_model)


def test_enrich_ollama(ollama_config):
    """Test the Ollama textual enrichment class for node enrichment."""
    # Prepare enrichment model
    cfg = ollama_config
    enr_model = EnrichmentWithOllama(
        model_name=cfg["model_name"],
        prompt_enrichment=cfg["prompt_enrichment"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
    )

    # Perform enrichment for nodes
    nodes = ["acetaminophen"]
    enriched_nodes = enr_model.enrich_documents(nodes)
    # Check the enriched nodes
    assert len(enriched_nodes) == 1
    assert all(enriched_nodes[i] != nodes[i] for i in range(len(nodes)))


def test_enrich_ollama_rag(ollama_config):
    """Test the Ollama textual enrichment class for enrichment with RAG (not implemented)."""
    # Prepare enrichment model
    cfg = ollama_config
    enr_model = EnrichmentWithOllama(
        model_name=cfg["model_name"],
        prompt_enrichment=cfg["prompt_enrichment"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
    )
    # Perform enrichment for nodes
    nodes = ["acetaminophen"]
    docs = [r"\path\to\doc1", r"\path\to\doc2"]
    enriched_nodes = enr_model.enrich_documents_with_rag(nodes, docs)
    # Check the enriched nodes
    assert len(enriched_nodes) == 1
    assert all(enriched_nodes[i] != nodes[i] for i in range(len(nodes)))
