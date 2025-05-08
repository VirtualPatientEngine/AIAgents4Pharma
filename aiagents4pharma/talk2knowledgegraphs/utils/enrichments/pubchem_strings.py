#!/usr/bin/env python3

"""
Enrichment class for enriching PubChem IDs with their STRINGS representation and descriptions.
"""

from typing import List
import pubchempy as pcp
from .enrichments import Enrichments
from ..pubchem_utils import pubchem_cid_description

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

        pubchem_cids = texts
        for pubchem_cid in pubchem_cids:
            try:
                c = pcp.Compound.from_cid(pubchem_cid)
            except pcp.BadRequestError:
                enriched_pubchem_ids_smiles.append(None)
                enriched_pubchem_ids_descriptions.append(None)
                continue
            enriched_pubchem_ids_smiles.append(c.isomeric_smiles)
            enriched_pubchem_ids_descriptions.append(pubchem_cid_description(pubchem_cid))

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
