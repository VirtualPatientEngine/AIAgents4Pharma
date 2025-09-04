"""
Test cases for Talk2Biomodels.
"""

from ..api.ols import fetch_from_ols
from ..api.uniprot import search_uniprot_labels


def test_search_uniprot_labels():
    """
    Test the search_uniprot_labels function.
    """
    # "P61764" = Positive result, "P0000Q" = negative result
    identifiers = ["P61764", "P0000Q"]
    results = search_uniprot_labels(identifiers)
    assert results["P61764"] == "Syntaxin-binding protein 1"
    assert results["P0000Q"].startswith("Error: 400")


def test_fetch_from_ols():
    """
    Test the fetch_from_ols function.
    """
    term_1 = "GO:0005886"  # Positive result
    term_2 = "GO:ABC123"  # Negative result
    label_1 = fetch_from_ols(term_1)
    label_2 = fetch_from_ols(term_2)
    assert isinstance(label_1, str), f"Expected string, got {type(label_1)}"
    assert isinstance(label_2, str), f"Expected string, got {type(label_2)}"
    assert label_1 == "plasma membrane"
    assert label_2.startswith("Error: 404")
