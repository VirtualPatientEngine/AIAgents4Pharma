#!/usr/bin/env python3

"""
Enrichment class for enriching PubChem IDs with their STRINGS representation.
"""

import logging
import requests
import hydra

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def drugbank_id2pubchem_cid(drugbank_id):
    """
    Convert DrugBank ID to PubChem CID.

    Args:
        drugbank_id: The DrugBank ID of the drug.

    Returns:
        The PubChem CID of the drug.
    """
    logger.log(logging.INFO, "Load Hydra configuration for PubChem ID conversion.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['utils/pubchem_utils=default'])
        cfg = cfg.utils.pubchem_utils
    # Prepare the URL
    pubchem_url_for_drug = cfg.drugbank_id_to_pubchem_cid_url + drugbank_id + '/JSON'
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
