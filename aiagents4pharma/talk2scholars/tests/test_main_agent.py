"""
Unit tests for the main agent functionality.
"""

from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage

from ..agents.main_agent import make_supervisor_node
from ..state.state_talk2scholars import Talk2Scholars

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
