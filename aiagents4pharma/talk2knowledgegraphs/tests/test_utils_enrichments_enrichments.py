"""
Test cases for utils/enrichments/enrichments.py
"""

from ..utils.enrichments.enrichments import Enrichments

class TestEnrichments(Enrichments):
    """Test implementation of the Enrichments interface for testing purposes."""

    def enrich_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            f"Additional text description of {text} as the input." for text in texts
        ]

    def enrich_documents_with_rag(self, texts, docs):
        # Currently we don't have a RAG model to test this method.
        # Thus, we will just call the enrich_documents method instead.
        return self.enrich_documents(texts)

def test_enrich_documents():
    """Test enriching documents using the Enrichments interface."""
    enrichments = TestEnrichments()
    texts = ["text1", "text2"]
    result = enrichments.enrich_documents(texts)
    assert result == [
        "Additional text description of text1 as the input.",
        "Additional text description of text2 as the input.",
    ]

def test_enrich_documents_with_rag():
    """Test enriching documents with RAG using the Enrichments interface."""
    enrichments = TestEnrichments()
    texts = ["text1", "text2"]
    docs = ["doc1", "doc2"]
    result = enrichments.enrich_documents_with_rag(texts, docs)
    assert result == [
        "Additional text description of text1 as the input.",
        "Additional text description of text2 as the input.",
    ]
