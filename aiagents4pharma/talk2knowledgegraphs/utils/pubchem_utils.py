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

def cas_rn2pubchem_cid(casrn):
    """
    Convert CAS RN to PubChem CID.

    Args:
        casrn: The CAS RN of the drug.

    Returns:
        The PubChem CID of the drug.
    """
    # Load Hydra configuration for PubChem ID conversion
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['utils/pubchem_utils=default'])
        cfg = cfg.utils.pubchem_utils
    # Prepare the URL
    pubchem_url_for_drug = f"{cfg.pubchem_casrn2cid_url}{casrn}/record/JSON"
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
        if cid is not None:
            break
    return cid

def external_id2pubchem_cid(db, db_id):
    """
    Convert external DB ID to PubChem CID.
    Please refer to the following URL for more information
    on data sources:
    https://pubchem.ncbi.nlm.nih.gov/sources/

    Args:
        db: The database name.
        db_id: The database ID of the drug.

    Returns:
        The PubChem CID of the drug.
    """
    # Load Hydra configuration for PubChem ID conversion
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['utils/pubchem_utils=default'])
        cfg = cfg.utils.pubchem_utils
    # Prepare the URL
    pubchem_url_for_drug = f"{cfg.pubchem_cid_base_url}/{db}/{db_id}/JSON"
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

def pubchem_cid_description(cid):
    """
    Get the description of a PubChem CID.

    Args:
        cid: The PubChem CID of the drug.

    Returns:
        The description of the PubChem CID.
    """
    # Load Hydra configuration for PubChem CID description
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['utils/pubchem_utils=default'])
        cfg = cfg.utils.pubchem_utils
    # Prepare the URL
    pubchem_url_for_descpription = f"{cfg.pubchem_cid_description_url}/{cid}/description/JSON"
    # Get the data
    response = requests.get(pubchem_url_for_descpription, timeout=60)
    data = response.json()
    # Extract the PubChem CID description
    description = ''
    for information in data["InformationList"]['Information']:
        description += information.get("Description", '')
    return description
