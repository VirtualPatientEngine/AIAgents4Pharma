#!/usr/bin/env python3

"""
Enrichment class for enriching Reactome pathways with textual descriptions
"""

from typing import List
import logging
import hydra
import requests
from .enrichments import Enrichments

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichmentWithReactome(Enrichments):
    """
    Enrichment class using Reactome pathways
    """
    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input Reactome pathways

        Args:
            texts: The list of Reactome pathways to be enriched.

        Returns:
            The list of enriched descriptions
        """

        reactome_pathways_ids = texts

        logger.log(logging.INFO,
                   "Load Hydra configuration for reactome enrichment")
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['utils/enrichments/reactome_pathways=default'])
            cfg = cfg.utils.enrichments.reactome_pathways

        descriptions = []
        for reactome_pathway_id in reactome_pathways_ids:
            r = requests.get(cfg.base_url + reactome_pathway_id + '/summation',
                             headers={ "Accept" : "text/plain"},
                             timeout=cfg.timeout)
            # if the response is not ok
            if not r.ok:
                descriptions.append(None)
                continue
            response_body = r.text
            # if the response is ok
            descriptions.append(response_body.split('\t')[1])
        return descriptions

    def enrich_documents_with_rag(self, texts, docs):
        """
        Enrich a list of input Reactome pathways

        Args:
            texts: The list of Reactome pathways to be enriched.

        Returns:
            The list of enriched descriptions
        """
        return self.enrich_documents(texts)
