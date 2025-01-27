"""
Enrichments interface
"""

from abc import ABC, abstractmethod

class Enrichments(ABC):
    """Interface for enrichment models.

    This is an interface meant for implementing text enrichment models.

    Enrichment models are used to enrich node or relation features in a given knowledge graph.
    """

    @abstractmethod
    def enrich_documents(self, texts: list[str]) -> list[list[str]]:
        """Enrich documents.

        Args:
            texts: List of documents to enrich.

        Returns:
            List of enriched documents.
        """
