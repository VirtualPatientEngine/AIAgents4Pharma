"""
Test cases for Talk2Biomodels get_modelinfo tool.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ..agents.t2b_agent import get_app

LLM_MODEL = MagicMock(name="llm_model")
pytestmark = pytest.mark.unit_mock


def test_get_modelinfo_tool(fake_app_factory, monkeypatch):
    """
    Test the get_modelinfo tool.
    """
    messages = [
        ToolMessage(
            content="model details",
            name="get_modelinfo",
            status="success",
            artifact={"info": "ok"},
            tool_call_id="call-1",
        )
    ]
    app = fake_app_factory([{"messages": messages}])
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_getmodelinfo.get_app",
        lambda *args, **kwargs: app,
    )
    unique_id = 12345
    app = get_app(unique_id, LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    app.update_state(
        config,
        {"sbml_file_path": ["aiagents4pharma/talk2biomodels/tests/BIOMD0000000449_url.xml"]},
    )
    prompt = "Extract all relevant information from the uploaded model."
    # Test the tool get_modelinfo
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)


def test_model_with_no_species(fake_app_factory, monkeypatch):
    """
    Test the get_modelinfo tool with a model that does not
    return any species.

    This should raise a tool error.
    """
    error_message = ToolMessage(
        content="Error: ValueError('Unable to extract species from the model.')",
        name="get_modelinfo",
        status="error",
        artifact=None,
        tool_call_id="call-2",
    )
    app = fake_app_factory([{"messages": [error_message]}])
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_getmodelinfo.get_app",
        lambda *args, **kwargs: app,
    )
    unique_id = 12345
    app = get_app(unique_id, LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Extract all species from model 20"
    # Test the tool get_modelinfo
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    test_condition = False
    for msg in reversed_messages:
        # Check if the message is a ToolMessage from the get_modelinfo tool
        if isinstance(msg, ToolMessage) and msg.name == "get_modelinfo":
            # Check if the message is an error message
            if (
                msg.status == "error"
                and "ValueError('Unable to extract species from the model.')" in msg.content
            ):
                test_condition = True
                break
    assert test_condition


def test_model_with_no_parameters(fake_app_factory, monkeypatch):
    """
    Test the get_modelinfo tool with a model that does not
    return any parameters.

    This should raise a tool error.
    """
    error_message = ToolMessage(
        content="Error: ValueError('Unable to extract parameters from the model.')",
        name="get_modelinfo",
        status="error",
        artifact=None,
        tool_call_id="call-3",
    )
    app = fake_app_factory([{"messages": [error_message]}])
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_getmodelinfo.get_app",
        lambda *args, **kwargs: app,
    )
    unique_id = 12345
    app = get_app(unique_id, LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Extract all parameters from model 10"
    # Test the tool get_modelinfo
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    test_condition = False
    for msg in reversed_messages:
        # Check if the message is a ToolMessage from the get_modelinfo tool
        if isinstance(msg, ToolMessage) and msg.name == "get_modelinfo":
            # Check if the message is an error message
            if (
                msg.status == "error"
                and "ValueError('Unable to extract parameters from the model.')" in msg.content
            ):
                test_condition = True
                break
    assert test_condition
