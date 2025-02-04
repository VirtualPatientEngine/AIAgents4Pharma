"""
This module contains the API for fetching uniprot database
"""
from typing import List, Dict
import requests

def search_uniprot_labels(identifiers: List[str]) -> Dict[str, str]:
    """
    Fetch protein names or labels for a list of UniProt identifiers by making sequential requests.

    Args:
        identifiers (List[str]): A list of UniProt identifiers.

    Returns:
        Dict[str, str]: A mapping of UniProt identifiers to their protein names or error messages.
    """
    results = {}
    base_url = "https://www.uniprot.org/uniprot/"

    for identifier in identifiers:
        url = f"{base_url}{identifier}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            protein_name = (
                data.get('proteinDescription', {})
                .get('recommendedName', {})
                .get('fullName', {})
                .get('value', 'Name not found')
            )
            results[identifier] = protein_name
        except requests.exceptions.RequestException as e:
            results[identifier] = f"Error: {str(e)}"
    return results
