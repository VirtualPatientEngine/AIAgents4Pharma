#!/usr/bin/env python3

"""
Enrichment class for enriching OLS terms with textual descriptions
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

class EnrichmentWithOLS(Enrichments):
    """
    Enrichment class using OLS terms
    """
    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input OLS terms

        Args:
            texts: The list of OLS terms to be enriched.

        Returns:
            The list of enriched descriptions
        """

        ols_ids = texts

        logger.log(logging.INFO,
                   "Load Hydra configuration for OLS enrichments.")
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['utils/enrichments/uniprot_proteins=default'])
            cfg = cfg.utils.enrichments.uniprot_proteins


        descriptions = []
        for ols_id in ols_ids:
            # https://www.ebi.ac.uk/ols4/api/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FUBERON_0000011&lang=en
            base_url = 'https://www.ebi.ac.uk/ols4/api/terms'
            params = {
                'short_form': ols_id
            }
            r = requests.get(base_url,
                             headers={ "Accept" : "application/json"},
                             params=params,
                             timeout=cfg.timeout)
            # if the response is not ok
            print (r.ok)
            if not r.ok:
                descriptions.append(None)
                continue
            response_body = json.loads(r.text)
            # if the response body is empty
            if '_embedded' not in response_body:
                descriptions.append(None)
                continue
            description = response_body['_embedded']['terms'][0]['description']
            label = response_body['_embedded']['terms'][0]['label']
            if not description:
                descriptions.append(label)
            else:
                descriptions.append('\n'.join(description))
        return descriptions

    def enrich_documents_with_rag(self, texts, docs):
        """
        Enrich a list of input OLS terms

        Args:
            texts: The list of OLS to be enriched.

        Returns:
            The list of enriched descriptions
        """
        return self.enrich_documents(texts)
