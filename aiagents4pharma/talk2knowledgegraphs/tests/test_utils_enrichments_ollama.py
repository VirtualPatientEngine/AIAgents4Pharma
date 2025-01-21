"""
Test cases for utils/enrichments/ollama.py
"""

import time
import pytest
import ollama
from ..utils.enrichments.ollama import EnrichmentWithOllama

@pytest.fixture(name="ollama_config")
def fixture_ollama_config():
    """Return a dictionary with Ollama configuration."""
    return {
        "model_name": "llama3.1",
        "prompt_enrichment": """
            You are a helpful expert in biomedical knowledge graph analysis.
            Your role is to enrich the inputs (nodes or relations) using textual description.
            A node is represented as string, e.g., "ADAM17" in the input list, while a relation is
            represented as tuples, e.g., "(NOD2, gene causation disease, Crohn disease)".
            DO NOT mistake one for the other. If the input is a list of nodes, treat each node as
            a unique entity, and provide a description. If the input is a list of relations,
            treat each tuple in the relation list as a unique relation between nodes,
            and provide a description for each tuple.
            All provided information about the node or relations should be concise
            (a single sentence), informative, factual, and relevant in the biomedical domain.

            ! IMPORTANT ! Make sure that the output is in valid format and can be parsed as
            a list of dictionaries correctly and without any prepend information.
            DO NOT forget to close the brackets properly.
            KEEP the order consistent between the input and the output.
            See <example> for reference.

            <example>
            Input: ["ADAM17", "IL23R"]
            Output: [{{"desc" : "ADAM17 is a metalloproteinase involved in the shedding of
            membrane proteins and plays a role in inflammatory processes."}}, {{"desc":
            "IL23R is a receptor for interleukin-23, which is involved in inflammatory responses
            and has been linked to inflammatory bowel disease."}}]
            </example>

            <example>
            Input: ["(NOD2, gene causation disease, Crohn disease)", "(IL23R,
            gene causation disease, Inflammatory Bowel Disease)"]
            Output: [{{"desc" : "NOD2 is a gene that contributes to immune responses and has
            been implicated in Crohn's disease, particularly through genetic mutations linked to
            inflammation."}}, {{"desc" : "IL23R is a receptor gene that plays a role in
            immune system regulation and has been associated with susceptibility to
            Inflammatory Bowel Disease."}}]
            </example>

            Input: {input}
        """,
        "temperature": 0.0,
        "streaming": False,
    }

def test_no_model_ollama(ollama_config):
    """Test the case when the Ollama model is not available."""
    cfg = ollama_config
    cfg_model = "qwen2:0.5b" # Choose a small model

    # Delete the Ollama model
    try:
        ollama.delete(cfg_model)
        time.sleep(10)
    except ollama.ResponseError:
        pass

    # Check if the model is available
    with pytest.raises(
        ValueError, match=f"Error: Pulled {cfg_model} model and restarted Ollama server."
    ):
        EnrichmentWithOllama(
            model_name=cfg_model,
            prompt_enrichment=cfg["prompt_enrichment"],
            temperature=cfg["temperature"],
            streaming=cfg["streaming"],
        )

def test_enrich_documents_ollama(ollama_config):
    """Test the Ollama textual enrichment class."""
    # Prepare enrichment model
    cfg = ollama_config
    enr_model = EnrichmentWithOllama(
        model_name=cfg["model_name"],
        prompt_enrichment=cfg["prompt_enrichment"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
    )

    # Perform enrichment for nodes
    nodes = ["Adalimumab", "Infliximab"]
    enriched_nodes = enr_model.enrich_documents(nodes)
    # Check the enriched nodes
    assert len(enriched_nodes) == 2
    assert all(
        len(enriched_nodes[i]["desc"]) > len(nodes[i]) for i in range(len(nodes))
    )

    # Perform enrichment for relations
    relations = [
        "(NOD2, gene causation disease, Crohn disease)",
        "(IL23R, gene causation disease, Inflammatory Bowel Disease)",
    ]
    enriched_relations = enr_model.enrich_documents(relations)
    # Check the enriched relations
    assert len(enriched_relations) == 2
    assert all(
        len(enriched_relations[i]["desc"]) > len(relations[i])
        for i in range(len(relations))
    )

def test_enrich_query_ollama(ollama_config):
    """Test the Ollama textual enrichment class."""
    # Prepare enrichment model
    cfg = ollama_config
    enr_model = EnrichmentWithOllama(
        model_name=cfg["model_name"],
        prompt_enrichment=cfg["prompt_enrichment"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
    )

    # Perform enrichment for a single node
    node = "Adalimumab"
    enriched_node = enr_model.enrich_query(node)
    # Check the enriched node
    assert len(enriched_node[0]["desc"]) > len(node)

    # Perform enrichment for a single relation
    relation = "(IL23R, gene causation disease, Inflammatory Bowel Disease)"
    enriched_relation = enr_model.enrich_query(relation)
    # Check the enriched relation
    assert len(enriched_relation[0]["desc"]) > len(relation)
