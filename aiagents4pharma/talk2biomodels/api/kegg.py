# import requests
# import re
# from typing import List, Dict

# def fetch_from_api(base_url: str, query: str) -> str:
#     """Fetch data from the given API endpoint."""
#     response = requests.get(base_url + query, timeout=10)
#     response.raise_for_status()
#     return response.text

# def fetch_kegg_names(ids: List[str]) -> Dict[str, str]:
#     """Fetch the names of multiple KEGG entries using the KEGG REST API."""
#     if not ids:
#         return {}

#     base_url = "https://rest.kegg.jp/get/"
#     query = "+".join(ids)
#     entry_data = fetch_from_api(base_url, query)
#     entries = entry_data.split("///")
#     entry_name_map = {}

#     for entry in entries:
#         if entry.strip():
#             lines = entry.strip().split("\n")
#             entry_line = next((line for line in lines if line.startswith("ENTRY")), None)
#             name_line = next((line for line in lines if line.startswith("NAME")), None)

#             if entry_line and name_line:
#                 entry_id = entry_line.split()[1]
#                 cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', '',name_line.replace("NAME", "").strip())
#                 entry_name_map[entry_id] = cleaned_name

#     return entry_name_map

# def fetch_kegg_annotations(data: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
#     """Fetch KEGG entry descriptions grouped by database type."""
#     grouped_data = {}
#     for entry in data:
#         db_type = entry["Database"].lower()
#         grouped_data.setdefault(db_type, []).append(entry["Id"])

#     results = {}
#     for db_type, ids in grouped_data.items():
#         results[db_type] = fetch_kegg_names(ids)

#     return results

# def get_protein_name_or_label(data: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
#     """Fetch descriptions for KEGG-related identifiers."""
#     return fetch_kegg_annotations(data)
import requests
import re
from typing import List, Dict

def fetch_from_api(base_url: str, query: str) -> str:
    """Fetch data from the given API endpoint."""
    try:
        response = requests.get(base_url + query, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for query {query}: {e}")
        return ""

def fetch_kegg_names(ids: List[str], batch_size: int = 10) -> Dict[str, str]:
    """
    Fetch the names of multiple KEGG entries using the KEGG REST API.

    Args:
        ids (List[str]): List of KEGG compound IDs.
        batch_size (int): Maximum number of IDs to include in a single request.

    Returns:
        Dict[str, str]: A mapping of KEGG IDs to their names.
    """
    if not ids:
        return {}

    base_url = "https://rest.kegg.jp/get/"
    entry_name_map = {}

    # Process IDs in batches
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        query = "+".join(batch)
        entry_data = fetch_from_api(base_url, query)

        if entry_data:
            entries = entry_data.split("///")
            for entry in entries:
                if entry.strip():
                    lines = entry.strip().split("\n")
                    entry_line = next((line for line in lines if line.startswith("ENTRY")), None)
                    name_line = next((line for line in lines if line.startswith("NAME")), None)

                    if entry_line and name_line:
                        # entry_id = entry_line.split()[1]
                        # # Split multiple names in the NAME field and clean them
                        # names = [re.sub(r'[^a-zA-Z0-9\\s]', '', name).strip() for name in name_line.replace("NAME", "").strip().split(";")]
                        # entry_name_map[entry_id] = "; ".join(names)
                        entry_id = entry_line.split()[1]
                        # Split multiple names in the NAME field and clean them
                        names = [
                            re.sub(r'[^a-zA-Z0-9\s]', '', name).strip()
                            for name in name_line.replace("NAME", "").strip().split(";")
                        ]
                        # Join cleaned names into a single string
                        entry_name_map[entry_id] = " ".join(names)

    return entry_name_map

def fetch_kegg_annotations(data: List[Dict[str, str]], batch_size: int = 10) -> Dict[str, Dict[str, str]]:
    """Fetch KEGG entry descriptions grouped by database type."""
    grouped_data = {}
    for entry in data:
        db_type = entry["Database"].lower()
        grouped_data.setdefault(db_type, []).append(entry["Id"])

    results = {}
    for db_type, ids in grouped_data.items():
        results[db_type] = fetch_kegg_names(ids, batch_size=batch_size)

    return results

def get_protein_name_or_label(data: List[Dict[str, str]], batch_size: int = 10) -> Dict[str, Dict[str, str]]:
    """Fetch descriptions for KEGG-related identifiers."""
    return fetch_kegg_annotations(data, batch_size=batch_size)

