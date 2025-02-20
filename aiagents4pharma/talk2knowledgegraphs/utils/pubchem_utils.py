#!/usr/bin/env python3

"""
Enrichment class for enriching PubChem IDs with their STRINGS representation.
"""

import requests

PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/drugbank/'

def drugbank_id2pubchem_cid(drugbank_id):
    """
    Convert DrugBank ID to PubChem CID.

    Args:
        drugbank_id: The DrugBank ID of the drug.

    Returns:
        The PubChem CID of the drug.
    """
    # Prepare the URL
    pubchem_url_for_drug = PUBCHEM_BASE_URL + drugbank_id + '/JSON'
    # Get the data
    response = requests.get(pubchem_url_for_drug, timeout=60)
    data = response.json()
    # Extract the PubChem CID
    cid = None
    for substance in data.get("PC_Substances", []):
        for compound in substance.get("compound", []):
            if "id" in compound and "type" in compound["id"] and compound["id"]["type"] == 1:
                cid = compound["id"].get("id", {}).get("cid")
                break
    return cid
