#!/usr/bin/env python3

"""
Test cases for utils/enrichments/ols_terms.py
"""

import pytest
from ..utils.enrichments.ols_terms import EnrichmentWithOLS

# In this test, we will consider 5 examples:
# 1. CL_0000899: T-helper 17 cell (Cell Ontology)
# 2. GO_0046427: positive regulation of receptor signaling pathway via JAK-STAT (GO)
# 3. UBERON_0000004: nose (Uberon)
# 4. HP_0009739: Hypoplasia of the antihelix (Human Phenotype Ontology)
# 5. MONDO_0005011: Crohn disease (MONDO)
# 6. XYZ_0000000: Non-existing term (for testing error handling)

# The expected description for each term starts with:
CL_DESC = "CD4-positive, alpha-beta T cell"
GO_DESC = "Any process that activates or increases the frequency, rate or extent"
UBERON_DESC = "The olfactory organ of vertebrates, consisting of nares"
HP_DESC = "Developmental hypoplasia of the antihelix"
MONDO_DESC = "A gastrointestinal disorder characterized by chronic inflammation"

# The expected description for the non-existing term is None


@pytest.fixture(name="enrich_obj")
def fixture_uniprot_config():
    """Return a dictionary with the configuration for OLS enrichment."""
    return EnrichmentWithOLS()


def test_enrich_documents(enrich_obj):
    """Test the enrich_documents method."""
    ols_terms = [
        "CL_0000899",
        "GO_0046427",
        "UBERON_0000004",
        "HP_0009739",
        "MONDO_0005011",
        "XYZ_0000000",
    ]
    descriptions = enrich_obj.enrich_documents(ols_terms)
    assert CL_DESC in descriptions[0]
    assert GO_DESC in descriptions[1]
    assert UBERON_DESC in descriptions[2]
    assert HP_DESC in descriptions[3]
    assert MONDO_DESC in descriptions[4]
    assert descriptions[5] is None


def test_enrich_documents_with_rag(enrich_obj):
    """Test the enrich_documents_with_rag method."""
    ols_terms = [
        "CL_0000899",
        "GO_0046427",
        "UBERON_0000004",
        "HP_0009739",
        "MONDO_0005011",
        "XYZ_0000000",
    ]
    descriptions = enrich_obj.enrich_documents_with_rag(ols_terms, None)
    assert CL_DESC in descriptions[0]
    assert GO_DESC in descriptions[1]
    assert UBERON_DESC in descriptions[2]
    assert HP_DESC in descriptions[3]
    assert MONDO_DESC in descriptions[4]
    assert descriptions[5] is None
