'''
Test cases for Talk2Biomodels.
'''

from ..api.uniprot import search_uniprot_labels
from ..api.ols import fetch_from_ols
from ..api.kegg import fetch_kegg_names, fetch_from_api

def test_search_uniprot_labels():
    '''
    Test the search_uniprot_labels function.
    '''
    identifiers = ["P61764", "P0000Q"]
    results = search_uniprot_labels(identifiers)
    assert results["P61764"] == "Syntaxin-binding protein 1"
    assert results["P0000Q"].startswith("Error: 400")

def test_fetch_from_ols():
    '''
    Test the fetch_from_ols function.
    '''
    term = "GO:ABC123"
    label = fetch_from_ols(term)
    assert label.startswith("Error: 404")

def test_fetch_kegg_names():
    '''
    Test the fetch_kegg_names function.
    '''
    ids = ["C00001", "C00002"]
    results = fetch_kegg_names(ids)
    assert results["C00001"] == "H2O"
    assert results["C00002"] == "ATP"

    # Try with an empty list
    results = fetch_kegg_names([])
    assert not results

def test_fetch_from_api():
    '''
    Test the fetch_from_api function.
    '''
    base_url = "https://rest.kegg.jp/get/"
    query = "C00001"
    entry_data = fetch_from_api(base_url, query)
    assert entry_data.startswith("ENTRY       C00001")

    # Try with an invalid query
    query = "C0000Q"
    entry_data = fetch_from_api(base_url, query)
    assert not entry_data
