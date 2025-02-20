#!/usr/bin/env python3

"""
Enrichment class for enriching PubChem IDs with their STRINGS representation.
"""

from typing import List
import pubchempy as pcp
from .enrichments import Enrichments

class EnrichmentWithPubChem(Enrichments):
    """
    Enrichment class using PubChem
    """
    def __init__(self):
        """
        Initialize the EnrichmentWithPubChem class.

        Args:
            model_name: The name of the Ollama model to be used.
            prompt_enrichment: The prompt enrichment template.
            temperature: The temperature for the Ollama model.
            streaming: The streaming flag for the Ollama model.
        """

    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input PubChem IDs with their STRINGS representation.

        Args:
            texts: The list of pubchem IDs to be enriched.

        Returns:
            The list of enriched STRINGS
        """

        enriched_pubchem_ids = []
        pubchem_cids = texts
        for pubchem_cid in pubchem_cids:
            try:
                c = pcp.Compound.from_cid(pubchem_cid)
            except pcp.BadRequestError:
                enriched_pubchem_ids.append(None)
                continue
            enriched_pubchem_ids.append(c.isomeric_smiles)

        return enriched_pubchem_ids

    def enrich_documents_with_rag(self, texts, docs):
        """
        Enrich a list of input PubChem IDs with their STRINGS representation.

        Args:
            texts: The list of pubchem IDs to be enriched.
            docs: The list of documents to be enriched.

        Returns:
            The list of enriched STRINGS
        """
        return self.enrich_documents(texts)
