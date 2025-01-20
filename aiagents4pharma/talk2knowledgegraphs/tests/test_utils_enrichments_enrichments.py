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

    def enrich_query(self, text: str) -> list[float]:
        return f"Additional text description of {text} as the input."

def test_enrich_documents():
    """Test enriching documents using the Enrichments interface."""
    enrichments = TestEnrichments()
    texts = ["text1", "text2"]
    result = enrichments.enrich_documents(texts)
    assert result == [
        "Additional text description of text1 as the input.",
        "Additional text description of text2 as the input.",
    ]

def test_enrich_query():
    """Test enriching a query using the Enrichments interface."""
    enrichments = TestEnrichments()
    text = "query"
    result = enrichments.enrich_query(text)
    assert result == "Additional text description of query as the input."
