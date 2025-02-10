"""
Unit tests for S2 tools functionality.
"""

from unittest.mock import patch
import pytest

from ..state.state_talk2scholars import Talk2Scholars
from ..tools.s2.display_results import display_results
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations
from ..tools.s2.search import search_tool
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from .conftest import MOCK_SEARCH_RESPONSE, MOCK_STATE_PAPER

class TestS2Tools:
    """Unit tests for individual S2 tools"""

    def test_display_results_empty_state(self, initial_state: Talk2Scholars):
        """Verifies display_results tool raises error when state is empty"""
        state = initial_state
        with pytest.raises(ValueError,
                         match="No papers found in the state. Please trigger a search."):
            display_results.invoke(input={"state": state})

    def test_display_results_shows_papers(self, initial_state: Talk2Scholars):
        """Verifies display_results tool correctly returns papers from state"""
        state = initial_state
        state["papers"] = MOCK_STATE_PAPER
        state["multi_papers"] = {}
        result = display_results.invoke(input={"state": state})
        # Check that result contains both papers and multi_papers
        assert isinstance(result, dict)
        assert result['papers'] == MOCK_STATE_PAPER
        assert result['multi_papers'] == {}

    @patch("requests.get")
    def test_search_finds_papers(self, mock_get):
        """Verifies search tool finds and formats papers correctly"""
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            input={
                "query": "machine learning",
                "limit": 1,
                "tool_call_id": "test123",
                "id": "test123",
            }
        )

        assert "papers" in result.update
        assert "messages" in result.update
        papers = result.update["papers"]
        assert isinstance(papers, dict)
        assert len(papers) > 0
        paper = next(iter(papers.values()))
        assert paper["Title"] == "Machine Learning Basics"
        assert paper["Year"] == 2023

    @patch("requests.get")
    def test_search_finds_papers_with_year(self, mock_get):
        """Verifies search tool works with year parameter"""
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            input={
                "query": "machine learning",
                "limit": 1,
                "year": "2023-",
                "tool_call_id": "test123",
                "id": "test123",
            }
        )

        assert "papers" in result.update
        assert "messages" in result.update
        papers = result.update["papers"]
        assert isinstance(papers, dict)
        assert len(papers) > 0

    @patch("requests.get")
    def test_search_filters_invalid_papers(self, mock_get):
        """Verifies search tool properly filters papers without title or authors"""
        mock_response = {
            "data": [
                {
                    "paperId": "123",
                    "abstract": "An introduction to ML",
                    "year": 2023,
                    "citationCount": 100,
                    "url": "https://example.com/paper1",
                    # Missing title and authors
                },
                MOCK_SEARCH_RESPONSE["data"][0]  # This one is valid
            ]
        }
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            input={
                "query": "machine learning",
                "limit": 2,
                "tool_call_id": "test123",
                "id": "test123",
            }
        )

        assert "papers" in result.update
        papers = result.update["papers"]
        assert len(papers) == 1  # Only the valid paper should be included

    @patch("requests.get")
    def test_single_paper_rec_basic(self, mock_get):
        """Tests basic single paper recommendation functionality"""
        mock_get.return_value.json.return_value = {
            "recommendedPapers": [MOCK_SEARCH_RESPONSE["data"][0]]
        }
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations.invoke(
            input={
                "paper_id": "123",
                "limit": 1,
                "tool_call_id": "test123",
                "id": "test123",
            }
        )
        assert "papers" in result.update
        assert len(result.update["messages"]) == 1

    @patch("requests.get")
    def test_single_paper_rec_with_optional_params(self, mock_get):
        """Tests single paper recommendations with year parameter"""
        mock_get.return_value.json.return_value = {
            "recommendedPapers": [MOCK_SEARCH_RESPONSE["data"][0]]
        }
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations.invoke(
            input={
                "paper_id": "123",
                "limit": 1,
                "year": "2023-",
                "tool_call_id": "test123",
                "id": "test123",
            }
        )
        assert "papers" in result.update

    @patch("requests.post")
    def test_multi_paper_rec_basic(self, mock_post):
        """Tests basic multi-paper recommendation functionality"""
        mock_post.return_value.json.return_value = {
            "recommendedPapers": [MOCK_SEARCH_RESPONSE["data"][0]]
        }
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations.invoke(
            input={
                "paper_ids": ["123", "456"],
                "limit": 1,
                "tool_call_id": "test123",
                "id": "test123",
            }
        )
        assert "multi_papers" in result.update
        assert len(result.update["messages"]) == 1

    @patch("requests.post")
    def test_multi_paper_rec_with_optional_params(self, mock_post):
        """Tests multi-paper recommendations with all optional parameters"""
        mock_post.return_value.json.return_value = {
            "recommendedPapers": [MOCK_SEARCH_RESPONSE["data"][0]]
        }
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations.invoke(
            input={
                "paper_ids": ["123", "456"],
                "limit": 1,
                "year": "2023-",
                "tool_call_id": "test123",
                "id": "test123",
            }
        )
        assert "multi_papers" in result.update
        assert len(result.update["messages"]) == 1
