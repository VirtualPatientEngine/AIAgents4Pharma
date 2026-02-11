"""
Test cases for Talk2Biomodels.
"""

import tempfile
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from ..agents.t2b_agent import get_app

LLM_MODEL = MagicMock(name="llm_model")
pytestmark = pytest.mark.unit_mock


def test_save_model_tool(fake_app_factory, monkeypatch):
    """
    Test the save_model tool.
    """
    states = [
        {"model_as_string": ["model-1"], "messages": []},
        {"model_as_string": ["model-2"], "messages": []},
        {"model_as_string": ["model-3"], "messages": []},
        {"model_as_string": ["model-4"], "messages": []},
    ]
    app = fake_app_factory(states)
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_save_model.get_app",
        lambda *args, **kwargs: app,
    )
    unique_id = 123
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    # Simulate a model
    prompt = "Simulate model 64"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Save a model without simulating
    prompt = "Save the model"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Create a temporary directory to save the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save a model to the temporary directory
        prompt = f"Simulate model 64 and save it model at {temp_dir}"
        # Invoke the agent
        app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        current_state = app.get_state(config)
        assert current_state.values["model_as_string"][-1] is not None
    # Simulate and save a model in non-existing path
    prompt = "Simulate model 64 and then save the model at /xyz/"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
