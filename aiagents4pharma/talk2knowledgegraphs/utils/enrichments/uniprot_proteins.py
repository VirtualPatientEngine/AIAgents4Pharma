#!/usr/bin/env python3

"""
Enrichment class for enriching Gene names with their function and sequence using UniProt.
"""

from typing import List
import logging
import json
import hydra
import requests
from .enrichments import Enrichments

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichmentWithUniProt(Enrichments):
    """
    Enrichment class using UniProt
    """
    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input UniProt gene names with their function and sequence.

        Args:
            texts: The list of gene names to be enriched.

        Returns:
            The list of enriched functions and sequences
        """

        enriched_gene_names = texts

        logger.log(logging.INFO,
                   "Load Hydra configuration for Gene enrichment with description and sequence.")
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['utils/enrichments/uniprot_proteins=default'])
            cfg = cfg.utils.enrichments.uniprot_proteins


        descriptions = []
        sequences = []
        for gene in enriched_gene_names:
            params = {
                "reviewed": cfg.reviewed,
                "isoform": cfg.isoform,
                "exact_gene": gene,
                "organism": cfg.organism,
                # You can get the list of all available organisms here:
                # https://www.uniprot.org/help/taxonomy
            }

            r = requests.get(cfg.uniprot_url,
                             headers={ "Accept" : "application/json"},
                             params=params,
                             timeout=cfg.timeout)
            # if the response is not ok
            if not r.ok:
                descriptions.append(None)
                sequences.append(None)
                continue
            response_body = json.loads(r.text)
            # if the response body is empty
            if not response_body:
                descriptions.append(None)
                sequences.append(None)
                continue
            description = ''
            for comment in response_body[0]['comments']:
                if comment['type'] == 'FUNCTION':
                    for value in comment['text']:
                        description += value['value']
            sequence = response_body[0]['sequence']['sequence']
            descriptions.append(description)
            sequences.append(sequence)
        return descriptions, sequences

    def enrich_documents_with_rag(self, texts, docs):
        """
        Enrich a list of input UniProt gene names with their function and sequence.

        Args:
            texts: The list of gene names to be enriched.

        Returns:
            The list of enriched functions and sequences
        """
        return self.enrich_documents(texts)
