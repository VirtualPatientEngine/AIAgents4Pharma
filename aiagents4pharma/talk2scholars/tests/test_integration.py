"""
Integration tests for Talk2Scholars system.
"""

from unittest.mock import Mock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ..agents.main_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars
from .conftest import MOCK_SEARCH_RESPONSE

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
