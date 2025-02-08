"""
Unit and integration tests for Talk2Scholars system.
Each test focuses on a single, specific functionality.
Tests are deterministic and independent of each other.
"""

from unittest.mock import Mock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ..agents.main_agent import get_app, make_supervisor_node
from ..state.state_talk2scholars import replace_dict, Talk2Scholars
from ..tools.s2.display_results import display_results
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations
from ..tools.s2.search import search_tool
from ..tools.s2.single_paper_rec import get_single_paper_recommendations

# pylint: disable=redefined-outer-name


@pytest.fixture(autouse=True)
def hydra_setup():
    """Setup and cleanup Hydra for tests."""
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        yield


@pytest.fixture
def mock_cfg() -> DictConfig:
    """Create a mock configuration for testing."""
    config = {
        "agents": {
            "talk2scholars": {
                "main_agent": {
                    "state_modifier": "Test prompt for main agent",
                    "temperature": 0,
                },
                "s2_agent": {
                    "temperature": 0,
                    "s2_agent": "Test prompt for s2 agent",
                },
            }
        },
        "tools": {
            "search": {
                "api_endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
                "default_limit": 2,
                "api_fields": ["paperId", "title", "abstract", "year", "authors"],
            }
        },
    }
    return OmegaConf.create(config)


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


@pytest.fixture
def initial_state() -> Talk2Scholars:
    """Create a base state for tests"""
    return Talk2Scholars(
        messages=[],
        papers={},
        is_last_step=False,
        current_agent=None,
        llm_model="gpt-4o-mini",
        next="",
    )


class TestMainAgent:
    """Unit tests for main agent functionality"""

    def test_supervisor_routes_search_to_s2(
        self, initial_state: Talk2Scholars, mock_cfg
    ):
        """Verifies that search-related queries are routed to S2 agent"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="Search initiated")

        # Extract the main_agent config
        supervisor = make_supervisor_node(
            llm_mock, mock_cfg.agents.talk2scholars.main_agent
        )
        state = initial_state
        state["messages"] = [HumanMessage(content="search for papers")]

        result = supervisor(state)
        assert result.goto == "s2_agent"
        assert not result.update["is_last_step"]
        assert result.update["current_agent"] == "s2_agent"

    def test_supervisor_routes_general_to_end(
        self, initial_state: Talk2Scholars, mock_cfg
    ):
        """Verifies that non-search queries end the conversation"""
        llm_mock = Mock()
        llm_mock.invoke.return_value = AIMessage(content="General response")

        # Extract the main_agent config
        supervisor = make_supervisor_node(
            llm_mock, mock_cfg.agents.talk2scholars.main_agent
        )
        state = initial_state
        state["messages"] = [HumanMessage(content="What is ML?")]

        result = supervisor(state)
        assert result.goto == "__end__"
        assert result.update["is_last_step"]


class TestS2Tools:
    """Unit tests for individual S2 tools"""

    def test_display_results_shows_papers(self, initial_state: Talk2Scholars):
        """Verifies display_results tool correctly returns papers from state"""
        state = initial_state
        state["papers"] = MOCK_STATE_PAPER
        result = display_results.invoke(input={"state": state})
        assert result == MOCK_STATE_PAPER
        assert isinstance(result, dict)

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
        assert "papers" in result.update
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
        assert "papers" in result.update
        assert len(result.update["messages"]) == 1


def test_state_replace_dict():
    """Verifies state dictionary replacement works correctly"""
    existing = {"key1": "value1", "key2": "value2"}
    new = {"key3": "value3"}
    result = replace_dict(existing, new)
    assert result == new
    assert isinstance(result, dict)


@pytest.mark.integration
def test_end_to_end_search_workflow(initial_state: Talk2Scholars, mock_cfg):
    """Integration test: Complete search workflow"""
    with (
        patch("requests.get") as mock_get,
        patch("langchain_openai.ChatOpenAI") as mock_llm,
        patch("hydra.compose", return_value=mock_cfg),
        patch("hydra.initialize"),
    ):
        mock_get.return_value.json.return_value = MOCK_SEARCH_RESPONSE
        mock_get.return_value.status_code = 200

        llm_instance = Mock()
        llm_instance.invoke.return_value = AIMessage(content="Search completed")
        mock_llm.return_value = llm_instance

        app = get_app("test_integration")
        test_state = initial_state
        test_state["messages"] = [HumanMessage(content="search for ML papers")]

        config = {
            "configurable": {
                "thread_id": "test_integration",
                "checkpoint_ns": "test",
                "checkpoint_id": "test123",
            }
        }

        response = app.invoke(test_state, config)
        assert "papers" in response
        assert len(response["messages"]) > 0
