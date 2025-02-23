"""
Unit tests for S2 tools functionality.
"""

# pylint: disable=redefined-outer-name
from unittest.mock import patch
from unittest.mock import MagicMock
import pytest
from langgraph.types import Command
from ..tools.s2.display_results import (
    display_results,
    NoPapersFoundError as raised_error,
)
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations
from ..tools.s2.search import search_tool
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from ..tools.s2.query_results import query_results, NoPapersFoundError
from ..tools.s2.retrieve_semantic_scholar_paper_id import (
    retrieve_semantic_scholar_paper_id,
)


@pytest.fixture
def initial_state():
    """Provides an empty initial state for tests."""
    return {"papers": {}, "multi_papers": {}}


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

    def test_display_results_empty_state(self, initial_state):
        """Verifies display_results tool behavior when state is empty and raises an exception"""
        with pytest.raises(
            raised_error,
            match="No papers found. A search/rec needs to be performed first.",
        ):
            display_results.invoke({"state": initial_state, "tool_call_id": "test123"})

    def test_display_results_shows_papers(self, initial_state):
        """Verifies display_results tool correctly returns papers from state"""
        state = initial_state.copy()
        state["last_displayed_papers"] = "papers"
        state["papers"] = MOCK_STATE_PAPER

        result = display_results.invoke(
            input={"state": state, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)  # Expect a Command object
        assert isinstance(result.update, dict)  # Ensure update is a dictionary
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert (
            "1 papers found. Papers are attached as an artifact."
            in result.update["messages"][0].content
        )

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
                MOCK_SEARCH_RESPONSE["data"][0],  # This one is valid
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
            input={"paper_id": "123", "limit": 1, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
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
            input={"paper_ids": ["123", "456"], "limit": 1, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
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

    @patch("requests.get")
    def test_search_tool_finds_papers(self, mock_get):
        """Verifies search tool finds and formats papers correctly"""
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            input={"query": "machine learning", "limit": 1, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)  # Expect a Command object
        assert "papers" in result.update
        assert len(result.update["papers"]) > 0

    def test_query_results_empty_state(self, initial_state):
        """Tests query_results tool behavior when no papers are found."""
        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first.",
        ):
            query_results.invoke(
                {"question": "List all papers", "state": initial_state}
            )

    @patch(
        "aiagents4pharma.talk2scholars.tools.s2.query_results.create_pandas_dataframe_agent"
    )
    def test_query_results_with_papers(self, mock_create_agent, initial_state):
        """Tests querying papers when data is available."""
        state = initial_state.copy()
        state["last_displayed_papers"] = "papers"
        state["papers"] = MOCK_STATE_PAPER

        # Mock the dataframe agent instead of the LLM
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Mocked response"}

        mock_create_agent.return_value = (
            mock_agent  # Mock the function returning the agent
        )

        # Ensure that the output of query_results is correctly structured
        result = query_results.invoke({"question": "List all papers", "state": state})

        assert isinstance(result, str)  # Ensure output is a string
        assert result == "Mocked response"  # Validate the expected response

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

    def test_single_paper_rec_invalid_id(self):
        """Test single paper recommendation with an invalid ID."""
        with pytest.raises(ValueError, match="Invalid paper ID or API error."):
            get_single_paper_recommendations.invoke(
                input={"paper_id": "", "tool_call_id": "test123"}  # Empty ID case
            )

    @patch("requests.post")
    def test_multi_paper_rec_no_recommendations(self, mock_post):
        """Tests behavior when multi-paper recommendation API returns no results."""
        mock_post.return_value.json.return_value = {"recommendedPapers": []}
        mock_post.return_value.status_code = 200

        result = get_multi_paper_recommendations.invoke(
            input={"paper_ids": ["123", "456"], "limit": 1, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
        assert "messages" in result.update
        assert (
            "No recommendations found based on multiple papers."
            in result.update["messages"][0].content
        )

    @patch("requests.get")
    def test_search_no_results(self, mock_get):
        """Tests behavior when search API returns no results."""
        mock_get.return_value.json.return_value = {"data": []}
        mock_get.return_value.status_code = 200

        result = search_tool.invoke(
            input={"query": "nonexistent topic", "limit": 1, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
        assert "messages" in result.update
        assert "No papers found." in result.update["messages"][0].content

    @patch("requests.get")
    def test_single_paper_rec_no_recommendations(self, mock_get):
        """Tests behavior when single paper recommendation API returns no results."""
        mock_get.return_value.json.return_value = {"recommendedPapers": []}
        mock_get.return_value.status_code = 200

        result = get_single_paper_recommendations.invoke(
            input={"paper_id": "123", "limit": 1, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
        assert "messages" in result.update
        assert "No recommendations found for" in result.update["messages"][0].content
