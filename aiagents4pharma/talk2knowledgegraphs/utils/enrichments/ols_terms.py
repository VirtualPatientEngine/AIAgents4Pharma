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
                                overrides=['utils/enrichments/ols_terms=default'])
            cfg = cfg.utils.enrichments.ols_terms

        descriptions = []
        for ols_id in ols_ids:
            params = {
                'short_form': ols_id
            }
            r = requests.get(cfg.base_url,
                             headers={ "Accept" : "application/json"},
                             params=params,
                             timeout=cfg.timeout)
            response_body = json.loads(r.text)
            # if the response body is empty
            if '_embedded' not in response_body:
                descriptions.append(None)
                continue
            # Add the description to the list
            description = []
            for term in response_body['_embedded']['terms']:
                # If the term has a description, add it to the list
                description += term.get('description', [])
                # Add synonyms to the description
                description += term.get('synonyms', [])
                # Add the label to the description
                # Label is not provided as list, so we need to convert it to a list
                description += [term.get('label', [])]
            # Make unique the description
            description = list(set(description))
            # Join the description with new line
            description = '\n'.join(description)
            # Add the description to the list
            descriptions.append(description)
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
