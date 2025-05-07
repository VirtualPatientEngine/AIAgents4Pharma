#!/usr/bin/env python3

"""
Test cases for utils/enrichments/pubchem_strings.py
"""

import pytest
from ..utils.enrichments.uniprot_proteins import EnrichmentWithUniProt

# In this test, we will consider 2 examples:
# 1. Gene Name: TP53
# 2. Gene Name: TP5 (Incomplete; must return empty results)
# 2. Gene Name: XZ (Shorter than 3 characters; must return empty results)
# The expected description of TP53 startswith:
START_DESCP = "Multifunctional transcription factor"
# The expected description of TP53 startswith:
START_SEQ = "MEEPQSDPSV"
# The expected SMILES representation for the second PubChem ID is None.

@pytest.fixture(name="enrich_obj")
def fixture_pubchem_config():
    """Return a dictionary with the configuration for UniProt enrichment."""
    return EnrichmentWithUniProt()

def test_enrich_documents(enrich_obj):
    """Test the enrich_documents method."""
    pubchem_ids = ["TP53", "TP5", "XZ"]
    descriptions, sequences = enrich_obj.enrich_documents(pubchem_ids)
    assert descriptions[0].startswith(START_DESCP)
    assert sequences[0].startswith(START_SEQ)
    assert descriptions[1] is None
    assert sequences[1] is None
    assert descriptions[2] is None
    assert sequences[2] is None

def test_enrich_documents_with_rag(enrich_obj):
    """Test the enrich_documents_with_rag method."""
    pubchem_ids = ["TP53", "TP5", "XZ"]
    descriptions, sequences = enrich_obj.enrich_documents_with_rag(pubchem_ids, None)
    assert descriptions[0].startswith(START_DESCP)
    assert sequences[0].startswith(START_SEQ)
    assert descriptions[1] is None
    assert sequences[1] is None
    assert descriptions[2] is None
    assert sequences[2] is None
