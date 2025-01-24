# import os
# import sys
# import requests

# def fetch_from_uniprot(identifier: str) -> str:
#     """Fetch the protein name or label based on the UniProt identifier."""
#     url = f"https://www.uniprot.org/uniprot/{identifier}.json"
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
#         return data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Name not found')
#     except requests.exceptions.RequestException:
#         return "Error: Unable to fetch data from UniProt."
import requests
from typing import List, Dict

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
        except KeyError:
            results[identifier] = "Error: Unexpected response structure"

    return results