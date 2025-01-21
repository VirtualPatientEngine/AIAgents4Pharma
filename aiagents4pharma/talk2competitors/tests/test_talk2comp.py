"""Test cases for talk2comp agents and tools"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException  # Add ToolException import

from agents.main_agent import get_app
from agents.s2_agent import s2_agent
from tools.s2.multi_paper_rec import get_multi_paper_recommendations
from tools.s2.search import search_tool
from tools.s2.single_paper_rec import get_single_paper_recommendations

# Mock data for tests
MOCK_PAPER_RESPONSE = {
    "recommendedPapers": [{"paperId": "abc123", "title": "Test Paper"}]
}


@pytest.fixture
def mock_api_response():
    """Fixture for mocked API responses"""
    return MOCK_PAPER_RESPONSE


def test_main_agent_routing():
    """Test the main agent's routing capabilities"""
    unique_id = "test_12345"
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}

    # Test search routing
    prompt = "Find me papers about machine learning"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    assert response["current_agent"] == "s2_agent"
    assert isinstance(response["messages"][-1], AIMessage)


def test_s2_agent():
    """Test the S2 agent's functionality"""
    assert hasattr(s2_agent, "tools_agent")

    state = {
        "messages": [HumanMessage(content="Find papers about machine learning")],
        "papers": [],
        "is_last_step": False,
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_PAPER_RESPONSE

        response = s2_agent.invoke(state)
        assert "messages" in response
        assert "papers" in response


def test_search_tool():
    """Test the search papers tool"""
    query = "machine learning"
    tool_call_id = "test_123"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [MOCK_PAPER_RESPONSE]}

        response = search_tool.func(query=query, tool_call_id=tool_call_id, limit=2)

        assert "papers" in response
        assert "messages" in response
        assert isinstance(response["papers"], list)


def test_single_paper_rec():
    """Test single paper recommendations"""
    paper_id = "1234567890123456789012345678901234567890"
    tool_call_id = "test_123"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_PAPER_RESPONSE

        response = get_single_paper_recommendations.func(
            paper_id=paper_id, tool_call_id=tool_call_id, limit=2
        )

        assert response is not None
        assert "papers" in response.update


def test_multi_paper_rec():
    """Test multi paper recommendations"""
    paper_ids = [
        "1234567890123456789012345678901234567890",
        "0987654321098765432109876543210987654321",
    ]
    tool_call_id = "test_123"

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = MOCK_PAPER_RESPONSE

        response = get_multi_paper_recommendations.func(
            paper_ids=paper_ids, tool_call_id=tool_call_id, limit=2
        )

        assert response is not None
        assert "papers" in response.update


def test_error_handling():
    """Test error handling in tools"""
    # Test invalid paper ID format
    with pytest.raises((ValueError, ToolException)) as exc_info:
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 404
            get_single_paper_recommendations.func(
                paper_id="invalid_id", tool_call_id="test_123", limit=2
            )
    assert any(
        msg in str(exc_info.value)
        for msg in ["40-character hexadecimal", "Error getting recommendations"]
    )

    # Test empty paper IDs list for multi-paper recommendations
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 400
        with pytest.raises((ValueError, ToolException)) as exc_info:
            get_multi_paper_recommendations.func(
                paper_ids=[], tool_call_id="test_123", limit=2
            )
    assert "At least one paper ID must be provided" in str(exc_info.value)

    # Test too many paper IDs
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 400
        with pytest.raises(ValueError) as exc_info:
            get_multi_paper_recommendations.func(
                paper_ids=["a" * 40 for _ in range(11)],  # 11 IDs
                tool_call_id="test_123",
                limit=2,
            )
    assert "Maximum of 10 paper IDs allowed" in str(exc_info.value)
