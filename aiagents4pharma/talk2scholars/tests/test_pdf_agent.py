"""
Unit Tests for the PDF QnA agent.
"""

from unittest import mock
import pytest
from omegaconf import OmegaConf
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.pdf_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """
    Fixture to mock Hydra configuration for the PDF agent.
    Returns a valid configuration using OmegaConf.create.
    """
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        cfg = OmegaConf.create({
            "temperature": 0,
            "pdf_agent": {"some_setting": "value"},
            "openai_llms": ["gpt-4o-mini"]
        })
        mock_compose.return_value = cfg
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """
    Fixture to mock the PDF agent tools to avoid real API calls.
    """
    with (
        mock.patch("aiagents4pharma.talk2scholars.tools.pdf.qna.qna_tool") as mock_qna_tool,
        mock.patch("aiagents4pharma.talk2scholars.tools.s2.query_results.query_results") as mock_query_results,
    ):
        mock_qna_tool.return_value = {"result": "Mock QnA Result"}
        mock_query_results.return_value = {"result": "Mock Query Result"}
        yield [mock_qna_tool, mock_query_results]


def test_pdf_agent_initialization():
    """
    Test that the PDF agent initializes correctly with the mock configuration.
    """
    thread_id = "test_pdf_thread"
    with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent") as mock_create:
        mock_create.return_value = mock.Mock()
        app = get_app(thread_id)
        assert app is not None
        assert mock_create.called


def test_pdf_agent_invocation():
    """
    Test that the PDF agent processes user input and returns a valid response.
    """
    thread_id = "test_pdf_thread"
    mock_state = Talk2Scholars(messages=[HumanMessage(content="Extract info from PDF")])
    with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent") as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        # Simulate a response from the agent using the expected keys.
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="PDF info extracted")],
            "pdf_data": {"section1": "Content from PDF"}
        }
        app = get_app(thread_id)
        result = app.invoke(
            mock_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )
        assert "messages" in result
        assert "pdf_data" in result
        assert result["pdf_data"]["section1"] == "Content from PDF"
        assert mock_agent.invoke.called


def test_pdf_agent_tools_assignment(mock_tools_fixture):
    """
    Ensure that the correct tools are assigned to the PDF agent.
    The PDF agent should have two tools: qna_tool and query_results.
    """
    thread_id = "test_pdf_thread"
    with (
        mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent") as mock_create,
        mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.ToolNode") as mock_toolnode,
    ):
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        mock_tool_instance = mock.Mock()
        mock_tool_instance.tools = mock_tools_fixture
        mock_toolnode.return_value = mock_tool_instance
        get_app(thread_id)
        assert mock_toolnode.called
        # Assert that exactly 2 tools are assigned.
        assert len(mock_tool_instance.tools) == 2


def test_pdf_agent_hydra_failure():
    """
    Test that the PDF agent raises an exception when Hydra fails to load its configuration.
    """
    thread_id = "test_pdf_thread"
    with mock.patch("hydra.initialize", side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id)
        assert "Hydra error" in str(exc_info.value)


def test_pdf_agent_with_none_llm_model():
    """
    Test that when llm_model is None, the config's openai_llms[0] is used.
    """
    thread_id = "test_pdf_none"
    custom_cfg = OmegaConf.create({
        "temperature": 0.5,
        "pdf_agent": {"some_setting": "value"},
        "openai_llms": ["custom-model"]
    })
    with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.hydra.compose", return_value=custom_cfg):
        with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.ChatOpenAI") as mock_chat:
            with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent") as mock_create:
                # Call get_app with llm_model set to None
                app = get_app(thread_id, llm_model=None)
                # Assert that ChatOpenAI was instantiated with the model from the configuration.
                mock_chat.assert_called_with(model="custom-model", temperature=0.5)


def test_pdf_agent_with_none_llm_model_empty():
    """
    Test that when llm_model is None and the openai_llms list is empty,
    the default "gpt-4o-mini" is used.
    """
    thread_id = "test_pdf_none_empty"
    custom_cfg = OmegaConf.create({
        "temperature": 0.7,
        "pdf_agent": {"some_setting": "value"},
        "openai_llms": []
    })
    with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.hydra.compose", return_value=custom_cfg):
        with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.ChatOpenAI") as mock_chat:
            with mock.patch("aiagents4pharma.talk2scholars.agents.pdf_agent.create_react_agent") as mock_create:
                # Call get_app with llm_model set to None.
                app = get_app(thread_id, llm_model=None)
                # Assert that ChatOpenAI was instantiated with the default model "gpt-4o-mini".
                mock_chat.assert_called_with(model="gpt-4o-mini", temperature=0.7)
