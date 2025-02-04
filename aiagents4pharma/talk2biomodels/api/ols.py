"""
This module contains the API for fetching ols database
"""
from typing import List, Dict
import requests

def fetch_from_ols(term: str) -> str:
    """
    Fetch the label for a single term from OLS.

    Args:
        term (str): The term in the format "ONTOLOGY:TERM_ID".

    Returns:
        str: The label for the term or an error message.
    """
    try:
        ontology, _ = term.split(":")
        base_url = f"https://www.ebi.ac.uk/ols4/api/ontologies/{ontology.lower()}/terms"
        params = {"obo_id": term}
        response = requests.get(
            base_url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        label = '-'
        # Extract and return the label
        if "_embedded" in data and "terms" in data["_embedded"] \
             and len(data["_embedded"]["terms"]) > 0:
            label = data["_embedded"]["terms"][0].get("label", "Label not found")
        return label
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        return f"Error: {str(e)}"

def fetch_ols_labels(terms: List[str]) -> Dict[str, str]:
    """
    Fetch labels for multiple terms from OLS.

    Args:
        terms (List[str]): A list of terms in the format "ONTOLOGY:TERM_ID".

    Returns:
        Dict[str, str]: A mapping of term IDs to their labels or error messages.
    """
    results = {}
    for term in terms:
        results[term] = fetch_from_ols(term)
    return results

def search_ols_labels(data: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Fetch OLS annotations grouped by ontology type.

    Args:
        data (List[Dict[str, str]]): A list of dictionaries containing 'Id' and 'Database'.

    Returns:
        Dict[str, Dict[str, str]]: A mapping of ontology type to term labels.
    """
    grouped_data = {}
    for entry in data:
        ontology = entry["Database"].lower()
        grouped_data.setdefault(ontology, []).append(entry["Id"])

    results = {}
    for ontology, terms in grouped_data.items():
        results[ontology] = fetch_ols_labels(terms)

    return results
