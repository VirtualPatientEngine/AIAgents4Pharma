#!/usr/bin/env python3

"""
Test cases for utils/enrichments/pubchem_strings.py
"""

import pytest
from ..utils.enrichments.pubchem_strings import EnrichmentWithPubChem

# In this test, we will consider 2 examples:
# 1. PubChem ID: 5311000 (Alclometasone)
# 2. PubChem ID: 1X (Fake ID)
# The expected SMILES representation for the first PubChem ID is:
SMILES_FIRST = 'C[C@@H]1C[C@H]2[C@@H]3[C@@H](CC4=CC(=O)C=C[C@@]'
SMILES_FIRST += '4([C@H]3[C@H](C[C@@]2([C@]1(C(=O)CO)O)C)O)C)Cl'
# The expected SMILES representation for the second PubChem ID is None.

@pytest.fixture(name="enrich_obj")
def fixture_pubchem_config():
    """Return a dictionary with the configuration for the PubChem enrichment."""
    return EnrichmentWithPubChem()

def test_enrich_documents(enrich_obj):
    """Test the enrich_documents method."""
    pubchem_ids = ["5311000", "1X"]
    enriched_strings = enrich_obj.enrich_documents(pubchem_ids)
    assert enriched_strings == [SMILES_FIRST, None]

def test_enrich_documents_with_rag(enrich_obj):
    """Test the enrich_documents_with_rag method."""
    pubchem_ids = ["5311000", "1X"]
    enriched_strings = enrich_obj.enrich_documents_with_rag(pubchem_ids, None)
    assert enriched_strings == [SMILES_FIRST, None]
