# """Unit tests for the main agent functionality."""
#
# from unittest.mock import Mock, patch
# import pytest
# from langchain_core.messages import AIMessage, HumanMessage
# from omegaconf import OmegaConf
# from langgraph.types import Command
# from ..state.state_talk2scholars import Talk2Scholars
# from ..agents.main_agent import get_app, make_supervisor_node
#
#
# class TestMainAgent:
#     """Unit tests for main agent functionality"""
#
#     @pytest.fixture
#     def mock_config(self):
#         """Create a mock configuration for testing."""
#         config = {
#             "agents": {
#                 "talk2scholars": {
#                     "main_agent": {
#                         "temperature": 0,
#                         "state_modifier": "You are an intelligent research assistant",
#                     }
#                 }
#             }
#         }
#         return OmegaConf.create(config)
#
#     @pytest.fixture
#     def test_state(self):
#         """Create a base test state"""
#         return Talk2Scholars(
#             messages=[],
#             papers={},
#             multi_papers={},
#             is_last_step=False,
#             current_agent=None,
#             llm_model="gpt-4o-mini",
#             next="s2_agent",
#             thread_id="test_thread",
#             need_search=False,
#         )
#
#     def test_supervisor_routes_search_to_s2(self, test_state):
#         """Test searching papers route to s2 agent"""
#
#         # Mock the supervisor node directly
#         def mock_supervisor(state):
#             return Command(
#                 goto="s2_agent",
#                 update={
#                     "messages": state["messages"],
#                     "is_last_step": False,
#                     "current_agent": "s2_agent",
#                     "papers": {},
#                     "multi_papers": {},
#                     "thread_id": "test_thread",
#                     "need_search": False,
#                 },
#             )
#
#         with patch(
#             "aiagents4pharma.talk2scholars.agents.main_agent.make_supervisor_node",
#             return_value=mock_supervisor,
#         ):
#             app = get_app("test_thread")
#
#             # Update test state with query
#             test_state.messages = [HumanMessage(content="search for papers about AI")]
#
#             # Run the workflow
#             response = app.invoke(test_state)
#
#             # Verify response
#             assert response.get("current_agent") == "s2_agent"
#             assert not response.get("is_last_step", True)
#
#     def test_supervisor_routes_general_to_end(self, test_state):
#         """Test general queries route to end"""
#
#         # Mock the supervisor node directly
#         def mock_supervisor(state):
#             return Command(
#                 goto="__end__",
#                 update={
#                     "messages": state["messages"]
#                     + [AIMessage(content="General response")],
#                     "is_last_step": True,
#                     "current_agent": None,
#                     "papers": {},
#                     "multi_papers": {},
#                     "need_search": False,
#                 },
#             )
#
#         with patch(
#             "aiagents4pharma.talk2scholars.agents.main_agent.make_supervisor_node",
#             return_value=mock_supervisor,
#         ):
#             app = get_app("test_thread")
#
#             # Update test state with query
#             test_state.messages = [HumanMessage(content="What is ML?")]
#
#             # Run the workflow
#             response = app.invoke(test_state)
#
#             # Verify response
#             assert len(response["messages"]) > 0
#             assert response.get("is_last_step", True)
#
#     def test_get_app_initialization(self, mock_config):
#         """Test app initialization"""
#         with (
#             patch("langchain_openai.ChatOpenAI"),
#             patch("hydra.compose", return_value=mock_config),
#             patch("hydra.initialize"),
#             patch("langgraph.prebuilt.create_react_agent"),
#         ):
#
#             app = get_app("test_thread", "gpt-4o-mini")
#             assert app is not None
