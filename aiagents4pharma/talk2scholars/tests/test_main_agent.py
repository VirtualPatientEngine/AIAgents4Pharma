"""
Unit tests for main agent functionality.
Tests the supervisor agent's routing logic and state management.
"""

# pylint: disable=redefined-outer-name,too-few-public-methods

from types import SimpleNamespace
import pytest
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import Field
from aiagents4pharma.talk2scholars.agents.main_agent import get_app

# --- Dummy LLM Implementation ---


class DummyLLM(BaseChatModel):
    """A dummy language model implementation for testing purposes."""

    model_name: str = Field(...)

    def _generate(self, prompt, stop=None):
        """Generate a response given a prompt."""
        DummyLLM.called_prompt = prompt
        return "dummy output"

    @property
    def _llm_type(self):
        """Return the type of the language model."""
        return "dummy"


# --- Dummy Workflow and Sub-agent Functions ---


class DummyWorkflow:
    """A dummy workflow class that records arguments for verification."""

    def __init__(self, supervisor_args=None):
        """Initialize the workflow with the given supervisor arguments."""
        self.supervisor_args = supervisor_args or {}
        self.checkpointer = None
        self.name = None

    def compile(self, checkpointer, name):
        """Compile the workflow with the given checkpointer and name."""
        self.checkpointer = checkpointer
        self.name = name
        return self


def dummy_s2_agent(uniq_id, llm_model):
    """Return a DummyWorkflow for the S2 agent."""
    dummy_s2_agent.called_uniq_id = uniq_id
    dummy_s2_agent.called_llm_model = llm_model
    return DummyWorkflow(supervisor_args={"agent": "s2", "uniq_id": uniq_id})


def dummy_zotero_agent(uniq_id, llm_model):
    """Return a DummyWorkflow for the Zotero agent."""
    dummy_zotero_agent.called_uniq_id = uniq_id
    dummy_zotero_agent.called_llm_model = llm_model
    return DummyWorkflow(supervisor_args={"agent": "zotero", "uniq_id": uniq_id})


def dummy_question_and_answer_agent(uniq_id, llm_model):
    """Return a DummyWorkflow for the PDF agent."""
    dummy_question_and_answer_agent.called_uniq_id = uniq_id
    dummy_question_and_answer_agent.called_llm_model = llm_model
    return DummyWorkflow(supervisor_args={"agent": "pdf", "uniq_id": uniq_id})


def dummy_create_supervisor(apps, model, state_schema, **kwargs):
    """Return a DummyWorkflow for the supervisor."""
    dummy_create_supervisor.called_kwargs = kwargs
    return DummyWorkflow(
        supervisor_args={
            "apps": apps,
            "model": model,
            "state_schema": state_schema,
            **kwargs,
        }
    )


# --- Dummy Hydra Configuration Setup ---


class DummyHydraContext:
    """A dummy context manager for mocking Hydra's initialize and compose functions."""

    def __enter__(self):
        """Return None when entering the context."""
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        """Exit function that does nothing."""
        return None


def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace object."""
    return SimpleNamespace(
        **{
            key: dict_to_namespace(val) if isinstance(val, dict) else val
            for key, val in d.items()
        }
    )


dummy_config = {
    "agents": {
        "talk2scholars": {"main_agent": {"system_prompt": "Dummy system prompt"}}
    }
}


class DummyHydraCompose:
    """A dummy class that returns a namespace from a dummy config dictionary."""

    def __init__(self, config):
        """Constructor that stores the dummy config."""
        self.config = config

    def __getattr__(self, item):
        """Return a namespace from the dummy config."""
        return dict_to_namespace(self.config.get(item, {}))


# --- Pytest Fixtures to Patch Dependencies ---


@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    """Patch the hydra.initialize and hydra.compose functions to return dummy objects."""
    monkeypatch.setattr(
        hydra, "initialize", lambda version_base, config_path: DummyHydraContext()
    )
    monkeypatch.setattr(
        hydra, "compose", lambda config_name, overrides: DummyHydraCompose(dummy_config)
    )


def dummy_paper_download_agent(uniq_id, llm_model):
    """Return a DummyWorkflow for the paper download agent."""
    dummy_paper_download_agent.called_uniq_id = uniq_id
    dummy_paper_download_agent.called_llm_model = llm_model
    return DummyWorkflow(
        supervisor_args={"agent": "paper_download", "uniq_id": uniq_id}
    )


@pytest.fixture(autouse=True)
def patch_sub_agents_and_supervisor(monkeypatch):
    """Patch the sub-agents and supervisor creation functions."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.get_app_s2", dummy_s2_agent
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.get_app_zotero",
        dummy_zotero_agent,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.get_app_pdf",
        dummy_question_and_answer_agent,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.get_app_paper_download",
        dummy_paper_download_agent,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.create_supervisor",
        dummy_create_supervisor,
    )


# --- Test Cases ---


def test_dummy_llm_generate():
    """Test the dummy LLM's generate function."""
    dummy = DummyLLM(model_name="test-model")
    output = getattr(dummy, "_generate")("any prompt")
    assert output == "dummy output"


def test_dummy_llm_llm_type():
    """Test the dummy LLM's _llm_type property."""
    dummy = DummyLLM(model_name="test-model")
    assert getattr(dummy, "_llm_type") == "dummy"


def test_get_app_with_gpt4o_mini():
    """
    Test that get_app replaces a 'gpt-4o-mini' LLM with a new ChatOpenAI instance.
    """
    uniq_id = "test_thread"
    dummy_llm = DummyLLM(model_name="gpt-4o-mini")
    app = get_app(uniq_id, dummy_llm)

    supervisor_args = getattr(app, "supervisor_args", {})
    assert isinstance(supervisor_args.get("model"), ChatOpenAI)
    assert supervisor_args.get("prompt") == "Dummy system prompt"
    assert getattr(app, "name", "") == "Talk2Scholars_MainAgent"


def test_get_app_with_other_model():
    """
    Test that get_app does not replace the LLM if its model_name is not 'gpt-4o-mini'.
    """
    uniq_id = "test_thread_2"
    dummy_llm = DummyLLM(model_name="other-model")
    app = get_app(uniq_id, dummy_llm)

    supervisor_args = getattr(app, "supervisor_args", {})
    assert supervisor_args.get("model") is dummy_llm
    assert supervisor_args.get("prompt") == "Dummy system prompt"
    assert getattr(app, "name", "") == "Talk2Scholars_MainAgent"
