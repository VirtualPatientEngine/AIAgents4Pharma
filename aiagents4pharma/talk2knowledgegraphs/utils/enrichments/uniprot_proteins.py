#!/usr/bin/env python3

"""
Enrichment class for enriching PubChem IDs with their STRINGS representation.
"""

from typing import List
import json
import requests
from .enrichments import Enrichments

# def remove_pubmed_references(text):
#     # Remove full parentheses that contain only PubMed references
#     text = re.sub(r'\((?:\s*PubMed:\d{7,8}\s*,?)*\)', '', text)
#     # Clean up any leftover individual PubMed references not in parentheses (just in case)
#     text = re.sub(r'PubMed:\d{7,8}', '', text)
#     # Normalize spacing
#     text = re.sub(r'\s{2,}', ' ', text).strip()
#     return text

class EnrichmentWithUniProt(Enrichments):
    """
    Enrichment class using UniProt
    """
    def enrich_documents(self, texts: List[str]) -> List[str]:
        """
        Enrich a list of input UniProt IDs with their function and sequence.

        Args:
            texts: The list of gene names to be enriched.

        Returns:
            The list of enriched functions and sequences
        """

        enriched_gene_names = texts

        request_url = "https://www.ebi.ac.uk/proteins/api/proteins"

        descriptions = []
        sequences = []
        for gene in enriched_gene_names:
            params = {
                "offset": 0,
                "size": 100,
                "reviewed": "true",
                "isoform": 0,
                "exact_gene": gene,
                "organism": "Homo sapiens"
            }

            r = requests.get(request_url,
                             headers={ "Accept" : "application/json"},
                             params=params,
                             timeout=5)
            if not r.ok:
                # r.raise_for_status()
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
        Enrich a list of input PubChem IDs with their STRINGS representation.

        Args:
            texts: The list of pubchem IDs to be enriched.
            docs: None

        Returns:
            The list of enriched STRINGS
        """
        return self.enrich_documents(texts)
