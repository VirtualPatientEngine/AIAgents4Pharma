#!/usr/bin/env python3

"""
Test cases for utils/enrichments/reactome_pathways.py
"""

import pytest
from ..utils.enrichments.reactome_pathways import EnrichmentWithReactome

# In this test, we will consider 2 examples:
# 1. R-HSA-3244647: cGAS binds cytosolic DNA
# 2. R-HSA-9905952: ATP binds P2RX7 in P2RX7 trimer:PANX1 heptamer
# 3. R-HSA-1234567: Fake pathway

# The expected description of pathway R-HSA-3244647 startswith:
FIRST_PATHWAY = "Cyclic GMP-AMP (cGAMP) synthase (cGAS) was identified as a cytosolic DNA"
# The expected description of pathway R-HSA-9905952 startswith:
SECOND_PATHWAY = "The P2RX7 (P2X7, P2Z) trimer binds ATP,"
# The expected description of pathway R-HSA-1234567 is None.

@pytest.fixture(name="enrich_obj")
def fixture_uniprot_config():
    """Return a dictionary with the configuration for Reactome enrichment."""
    return EnrichmentWithReactome()

def test_enrich_documents(enrich_obj):
    """Test the enrich_documents method."""
    reactome_pathways = ["R-HSA-3244647",
                         "R-HSA-9905952",
                         "R-HSA-1234567"]
    descriptions = enrich_obj.enrich_documents(reactome_pathways)
    assert descriptions[0].startswith(FIRST_PATHWAY)
    assert descriptions[1].startswith(SECOND_PATHWAY)
    assert descriptions[2] is None

def test_enrich_documents_with_rag(enrich_obj):
    """Test the enrich_documents_with_rag method."""
    reactome_pathways = ["R-HSA-3244647",
                         "R-HSA-9905952",
                         "R-HSA-1234567"]
    descriptions = enrich_obj.enrich_documents_with_rag(reactome_pathways, None)
    assert descriptions[0].startswith(FIRST_PATHWAY)
    assert descriptions[1].startswith(SECOND_PATHWAY)
    assert descriptions[2] is None
