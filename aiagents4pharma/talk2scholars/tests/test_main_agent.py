"""
Unit tests for main agent functionality.
Tests the supervisor agent's routing logic and state management.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=redefined-outer-name,too-few-public-methods


import pytest
import hydra
import logging
from types import SimpleNamespace

# Import the function to test.
from aiagents4pharma.talk2scholars.agents.main_agent import get_app
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field

# --- Dummy LLM Implementation ---


class DummyLLM(BaseChatModel):
    model_name: str = Field(...)

    def _generate(self, prompt, stop=None):
        # Dummy implementation just returns a fixed string.
        return "dummy output"

    @property
    def _llm_type(self):
        return "dummy"


# --- Dummy Workflow and Sub-agent Functions ---


class DummyWorkflow:
    def __init__(self, supervisor_args=None):
        self.supervisor_args = supervisor_args or {}

    def compile(self, checkpointer, name):
        self.checkpointer = checkpointer
        self.name = name
        return self


def dummy_get_app_s2(uniq_id, llm_model):
    return DummyWorkflow(supervisor_args={"agent": "s2", "uniq_id": uniq_id})


def dummy_get_app_zotero(uniq_id, llm_model):
    return DummyWorkflow(supervisor_args={"agent": "zotero", "uniq_id": uniq_id})


def dummy_create_supervisor(
    apps, model, state_schema, output_mode, add_handoff_back_messages, prompt
):
    # Record arguments for later verification.
    supervisor_args = {
        "apps": apps,
        "model": model,
        "state_schema": state_schema,
        "output_mode": output_mode,
        "add_handoff_back_messages": add_handoff_back_messages,
        "prompt": prompt,
    }
    return DummyWorkflow(supervisor_args=supervisor_args)


# --- Dummy Hydra Configuration Setup ---


class DummyHydraContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        pass


def dict_to_namespace(d):
    ns = SimpleNamespace(
        **{
            key: dict_to_namespace(val) if isinstance(val, dict) else val
            for key, val in d.items()
        }
    )
    return ns


dummy_config = {
    "agents": {
        "talk2scholars": {"main_agent": {"system_prompt": "Dummy system prompt"}}
    }
}


class DummyHydraCompose:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, item):
        return dict_to_namespace(self.config.get(item, {}))


# --- Pytest Fixtures to Patch Dependencies ---


@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    # Patch hydra.initialize to return our dummy context.
    monkeypatch.setattr(
        hydra, "initialize", lambda version_base, config_path: DummyHydraContext()
    )
    # Patch hydra.compose to return our dummy config as a namespace.
    monkeypatch.setattr(
        hydra, "compose", lambda config_name, overrides: DummyHydraCompose(dummy_config)
    )


@pytest.fixture(autouse=True)
def patch_sub_agents_and_supervisor(monkeypatch):
    # Patch the sub-agent functions and supervisor creation in main_agent.
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.get_app_s2", dummy_get_app_s2
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.get_app_zotero",
        dummy_get_app_zotero,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.agents.main_agent.create_supervisor",
        dummy_create_supervisor,
    )


# --- Test Cases ---


def test_dummy_llm_generate():
    dummy = DummyLLM(model_name="test-model")
    # Calling _generate should return "dummy output"
    output = dummy._generate("any prompt")
    assert output == "dummy output"


def test_dummy_llm_llm_type():
    dummy = DummyLLM(model_name="test-model")
    # The _llm_type property should return "dummy"
    assert dummy._llm_type == "dummy"


def test_get_app_with_gpt4o_mini(monkeypatch):
    """
    Test that get_app replaces a 'gpt-4o-mini' LLM with a new ChatOpenAI instance,
    loads the Hydra config, and creates a supervisor workflow with the expected prompt.
    """
    uniq_id = "test_thread"
    # Create a dummy LLM with model_name "gpt-4o-mini" (using keyword argument)
    dummy_llm = DummyLLM(model_name="gpt-4o-mini")
    app = get_app(uniq_id, dummy_llm)
    # The tool should have replaced the LLM with a ChatOpenAI instance.
    supervisor_args = app.supervisor_args
    from langchain_openai import ChatOpenAI

    assert isinstance(supervisor_args["model"], ChatOpenAI)
    # The prompt should be taken from our dummy Hydra config.
    assert supervisor_args["prompt"] == "Dummy system prompt"
    # The compiled app should have the expected name.
    assert app.name == "Talk2Scholars_MainAgent"


def test_get_app_with_other_model(monkeypatch):
    """
    Test that get_app does not replace the LLM if its model_name is not 'gpt-4o-mini'
    and that the supervisor workflow receives the correct model.
    """
    uniq_id = "test_thread_2"
    dummy_llm = DummyLLM(model_name="other-model")
    app = get_app(uniq_id, dummy_llm)
    supervisor_args = app.supervisor_args
    # In this case, the model should be the dummy_llm since it was not replaced.
    assert supervisor_args["model"] is dummy_llm
    assert supervisor_args["prompt"] == "Dummy system prompt"
    assert app.name == "Talk2Scholars_MainAgent"
