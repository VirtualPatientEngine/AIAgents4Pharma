# """
# Integration tests for Talk2Scholars system.
# """
#
# from unittest.mock import Mock, patch
# import json
# import pytest
# from langchain_core.messages import AIMessage, HumanMessage
# from omegaconf import OmegaConf
#
# from ..agents.main_agent import get_app
# from ..state.state_talk2scholars import Talk2Scholars
#
#
# class TestIntegration:
#     """Integration tests for the Talk2Scholars system"""
#
#     @pytest.fixture
#     def mock_config(self):
#         """Create a mock configuration for testing."""
#         config = {
#             "agents": {
#                 "talk2scholars": {
#                     "main_agent": {
#                         "temperature": 0,
#                         "system": "You are an intelligent research assistant.",
#                     },
#                     "s2_agent": {
#                         "temperature": 0,
#                         "system": "You are a specialized academic research agent.",
#                     },
#                 }
#             },
#             "tools": {
#                 "search": {
#                     "api_endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
#                     "default_limit": 2,
#                     "request_timeout": 10,
#                     "api_fields": ["paperId", "title", "abstract", "year", "authors"],
#                 }
#             },
#         }
#         return OmegaConf.create(config)
#
#     @pytest.fixture
#     def initial_state(self):
#         """Create initial state for testing"""
#         return Talk2Scholars(
#             messages=[],
#             papers={},
#             multi_papers={},
#             is_last_step=False,
#             current_agent=None,
#             llm_model="gpt-4o-mini",
#             next="s2_agent",
#             thread_id="test_integration",
#             need_search=False,
#         )
#
#     @pytest.fixture
#     def mock_search_response(self):
#         """Mock search response data"""
#         return {
#             "data": [
#                 {
#                     "paperId": "123",
#                     "title": "Machine Learning Basics",
#                     "abstract": "An introduction to ML",
#                     "year": 2023,
#                     "citationCount": 100,
#                     "url": "https://example.com/paper1",
#                     "authors": [{"name": "Test Author"}],
#                 }
#             ]
#         }
#
#     @pytest.mark.integration
#     def test_end_to_end_search_workflow(
#         self, initial_state, mock_config, mock_search_response
#     ):
#         """Test complete search workflow from query to results"""
#         with (
#             patch("requests.get") as mock_get,
#             patch("langchain_openai.ChatOpenAI") as mock_llm,
#             patch("hydra.compose", return_value=mock_config),
#             patch("hydra.initialize"),
#         ):
#             # Setup mock responses
#             mock_get.return_value.json.return_value = mock_search_response
#             mock_get.return_value.status_code = 200
#
#             # Mock LLM responses
#             llm_instance = Mock()
#             llm_instance.invoke.side_effect = [
#                 # Supervisor response
#                 {"messages": [AIMessage(content=json.dumps({"next": "s2_agent"}))]},
#                 # S2 agent response
#                 {"messages": [AIMessage(content="Search completed successfully")]},
#             ]
#             mock_llm.return_value = llm_instance
#
#             # Initialize app
#             app = get_app("test_integration")
#
#             # Setup test state
#             test_state = initial_state
#             test_state["messages"] = [HumanMessage(content="search for ML papers")]
#
#             # Execute workflow
#             response = app.invoke(test_state)
#
#             # Verify response
#             assert response is not None
#             assert "papers" in response
#             assert len(response["messages"]) > 0
#             assert not response.get("is_last_step", True)
#
#     @pytest.mark.integration
#     def test_error_handling_workflow(self, initial_state, mock_config):
#         """Test workflow with API errors"""
#         with (
#             patch("requests.get") as mock_get,
#             patch("langchain_openai.ChatOpenAI") as mock_llm,
#             patch("hydra.compose", return_value=mock_config),
#             patch("hydra.initialize"),
#         ):
#             # Setup mock error response
#             mock_get.return_value.status_code = 500
#
#             # Mock LLM responses
#             llm_instance = Mock()
#             llm_instance.invoke.side_effect = [
#                 {"messages": [AIMessage(content=json.dumps({"next": "s2_agent"}))]},
#                 {"messages": [AIMessage(content="Error in search")]},
#             ]
#             mock_llm.return_value = llm_instance
#
#             # Initialize app
#             app = get_app("test_integration")
#
#             # Setup test state
#             test_state = initial_state
#             test_state["messages"] = [HumanMessage(content="search for papers")]
#
#             # Execute workflow
#             response = app.invoke(test_state)
#
#             # Verify error handling
#             assert response is not None
#             assert "messages" in response
#             assert len(response["messages"]) > 0
