#!/usr/bin/env python3

"""
Enrichment class for enriching PubChem IDs with their STRINGS representation and descriptions.
"""

import logging
from typing import List
import requests
import hydra
from .enrichments import Enrichments
from ..pubchem_utils import pubchem_cid_description

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichmentWithPubChem(Enrichments):
    """
    Enrichment class using PubChem
    """
    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input PubChem IDs with their STRINGS representation.

        Args:
            texts: The list of pubchem IDs to be enriched.

        Returns:
            The list of enriched STRINGS and their descriptions.
        """

        enriched_pubchem_ids_smiles = []
        enriched_pubchem_ids_descriptions = []

        # Load Hydra configuration to get the base URL for PubChem
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['utils/pubchem_utils=default'])
            cfg = cfg.utils.pubchem_utils
        # Iterate over each PubChem ID in the input list
        pubchem_cids = texts
        for pubchem_cid in pubchem_cids:
            # Prepare the URL
            pubchem_url = f"{cfg.pubchem_cid2smiles_url}/{pubchem_cid}/property/smiles/JSON"
            # Get the data
            response = requests.get(pubchem_url, timeout=60)
            data = response.json()
            # Extract the PubChem CID SMILES
            smiles = ''
            description = ''
            if "PropertyTable" in data:
                for prop in data["PropertyTable"]['Properties']:
                    smiles = prop.get("SMILES", '')
                    description = pubchem_cid_description(pubchem_cid)
            else:
                # If the PubChem ID is not found, set smiles and description to None
                smiles = None
                description = None
            enriched_pubchem_ids_smiles.append(smiles)
            enriched_pubchem_ids_descriptions.append(description)

        return enriched_pubchem_ids_descriptions, enriched_pubchem_ids_smiles

    def enrich_documents_with_rag(self, texts, docs):
        """
        Enrich a list of input PubChem IDs with their STRINGS representation.

        Args:
            texts: The list of pubchem IDs to be enriched.
            docs: None

        Returns:
            The list of enriched STRINGS
        """
        return self.enrich_documents(texts)
