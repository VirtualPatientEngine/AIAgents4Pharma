#!/usr/bin/env python3

"""
Test cases for utils/enrichments/uniprot_proteins.py
"""

import pytest
from ..utils.enrichments.uniprot_proteins import EnrichmentWithUniProt

# In this test, we will consider 2 examples:
# 1. Gene Name: TP53
# 2. Gene Name: TP5 (Incomplete; must return empty results)
# 2. Gene Name: XZ (Shorter than 3 characters; must return empty results)
# The expected description of TP53 startswith:
START_DESCP = "Multifunctional transcription factor"
# The expected amino acid sequence of TP53 startswith:
START_SEQ = "MEEPQSDPSV"

@pytest.fixture(name="enrich_obj")
def fixture_uniprot_config():
    """Return a dictionary with the configuration for UniProt enrichment."""
    return EnrichmentWithUniProt()

def test_enrich_documents(enrich_obj):
    """Test the enrich_documents method."""
    gene_names = ["TP53", "TP5", "XZ"]
    descriptions, sequences = enrich_obj.enrich_documents(gene_names)
    assert descriptions[0].startswith(START_DESCP)
    assert sequences[0].startswith(START_SEQ)
    assert descriptions[1] is None
    assert sequences[1] is None
    assert descriptions[2] is None
    assert sequences[2] is None

def test_enrich_documents_with_rag(enrich_obj):
    """Test the enrich_documents_with_rag method."""
    gene_names = ["TP53", "TP5", "XZ"]
    descriptions, sequences = enrich_obj.enrich_documents_with_rag(gene_names, None)
    assert descriptions[0].startswith(START_DESCP)
    assert sequences[0].startswith(START_SEQ)
    assert descriptions[1] is None
    assert sequences[1] is None
    assert descriptions[2] is None
    assert sequences[2] is None
