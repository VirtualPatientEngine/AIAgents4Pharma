"""
Test cases for Talk2Biomodels.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ..agents.t2b_agent import get_app

LLM_MODEL = MagicMock(name="llm_model")
pytestmark = pytest.mark.unit_mock


def _build_app(fake_app_factory, monkeypatch, states):
    """Return app and config with patched get_app."""
    app = fake_app_factory(states)
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_integration.get_app",
        lambda *_args, **_kwargs: app,
    )
    unique_id = 1234567
    app = get_app(unique_id, llm_model=LLM_MODEL)
    return app, {"configurable": {"thread_id": unique_id}}


def test_integration_simulate(fake_app_factory, monkeypatch):
    """Test simulate_model path and assistant response."""
    simulate_message = ToolMessage(
        content="Simulation results ready",
        name="simulate_model",
        status="success",
        artifact={"data": {"CRP{serum}": [1]}},
        tool_call_id="call-1",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [{"messages": [simulate_message]}])
    prompt = (
        "Simulate the model BIOMD0000000537 for 100 hours and time intervals "
        "100 with an initial concentration of `DoseQ2W` set to 300 and `Dose` "
        "set to 0. Reset the concentration of `Ab{serum}` to 100 every 25 hours."
    )
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    assert isinstance(response["messages"][-1].content, str)


def test_integration_ask_question(fake_app_factory, monkeypatch):
    """Test ask_question path when simulation results exist."""
    ask_question_message = ToolMessage(
        content="Value 211",
        name="ask_question",
        status="success",
        artifact=None,
        tool_call_id="call-2",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [{"messages": [ask_question_message]}])
    app.update_state(config, {"llm_model": LLM_MODEL})
    prompt = (
        "What is the concentration of CRP in serum after 100 hours? "
        "Round off the value to 2 decimal places."
    )
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    assert "211" in response["messages"][-1].content


def test_integration_custom_plotter_missing(fake_app_factory, monkeypatch):
    """Test custom_plotter error when species are missing."""
    missing_plot_message = ToolMessage(
        content="Species not found",
        name="custom_plotter",
        status="error",
        artifact=None,
        tool_call_id="call-3",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [{"messages": [missing_plot_message]}])
    app.update_state(config, {"llm_model": LLM_MODEL})
    prompt = (
        "Call the custom_plotter tool to make a plot showing only species "
        "'Infected cases'. Let me know if these species were not found. "
        "Do not invoke any other tool."
    )
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    predicted_artifact = None
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage) and msg.name == "custom_plotter":
            predicted_artifact = msg.artifact
            break
    assert predicted_artifact is None


def test_integration_custom_plotter_success(fake_app_factory, monkeypatch):
    """Test custom_plotter success and artifact header validation."""
    plot_message = ToolMessage(
        content="Plot generated",
        name="custom_plotter",
        status="success",
        artifact={
            "dic_data": [
                {
                    "Time": 0,
                    "CRP{serum}": 1,
                    "CRPExtracellular": 2,
                    "CRP Suppression (%)": 3,
                    "CRP (% of baseline)": 4,
                    "CRP{liver}": 5,
                }
            ]
        },
        tool_call_id="call-4",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [{"messages": [plot_message]}])
    app.update_state(config, {"llm_model": LLM_MODEL})
    prompt = "Plot only CRP related species."
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    predicted_artifact = []
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage) and msg.name == "custom_plotter":
            predicted_artifact = msg.artifact["dic_data"]
            break
    df = pd.DataFrame(predicted_artifact)
    predicted_header = df.columns.tolist()
    expected_header = [
        "Time",
        "CRP{serum}",
        "CRPExtracellular",
        "CRP Suppression (%)",
        "CRP (% of baseline)",
        "CRP{liver}",
    ]
    assert set(expected_header).issubset(set(predicted_header))
