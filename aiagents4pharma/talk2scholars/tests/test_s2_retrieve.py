"""
Unit tests for S2 tools functionality.
"""

# pylint: disable=redefined-outer-name
from unittest.mock import patch
import pytest
from langgraph.types import Command
from ..tools.s2.retrieve_semantic_scholar_paper_id import (
    retrieve_semantic_scholar_paper_id,
)


# Fixed test data for deterministic results
MOCK_SEARCH_RESPONSE = {
    "data": [
        {
            "paperId": "123",
            "title": "Machine Learning Basics",
            "abstract": "An introduction to ML",
            "year": 2023,
            "citationCount": 100,
            "url": "https://example.com/paper1",
            "authors": [{"name": "Test Author"}],
        }
    ]
}

MOCK_STATE_PAPER = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "URL": "https://example.com/paper1",
    }
}


class TestS2Tools:
    """Unit tests for individual S2 tools"""

    @patch("requests.get")
    def test_retrieve_semantic_scholar_paper_id(self, mock_get):
        """Tests retrieving a paper ID from Semantic Scholar."""
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_get.return_value.status_code = 200

        result = retrieve_semantic_scholar_paper_id.invoke(
            input={"paper_title": "Machine Learning Basics", "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
        assert "messages" in result.update
        assert (
            "Paper ID for 'Machine Learning Basics' is: 123"
            in result.update["messages"][0].content
        )

    def test_retrieve_semantic_scholar_paper_id_no_results(self):
        """Test retrieving a paper ID when no results are found."""
        with pytest.raises(ValueError, match="No papers found for query: UnknownPaper"):
            retrieve_semantic_scholar_paper_id.invoke(
                input={"paper_title": "UnknownPaper", "tool_call_id": "test123"}
            )
