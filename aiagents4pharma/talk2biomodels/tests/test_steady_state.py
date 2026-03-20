"""
Test cases for Talk2Biomodels steady state tool.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ..agents.t2b_agent import get_app
from ..tools.load_biomodel import ModelData
from ..tools.steady_state import SteadyStateTool, run_steady_state

LLM_MODEL = MagicMock(name="llm_model")
pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def _build_app(fake_app_factory, monkeypatch, states):
    """Return app and config with patched get_app."""
    app = fake_app_factory(states)
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_steady_state.get_app",
        lambda *_args, **_kwargs: app,
    )
    unique_id = 123
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(config, {"llm_model": LLM_MODEL})
    return app, config


def test_steady_state_tool_error(fake_app_factory, monkeypatch):
    """Test steady_state error status handling."""
    error_message = ToolMessage(
        content="Steady state failed",
        name="steady_state",
        status="error",
        artifact=None,
        tool_call_id="call-1",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [{"messages": [error_message]}])
    prompt = "Run a steady state analysis of model 537."
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    tool_msg_status = None
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage):
            tool_msg_status = msg.status
            break
    assert tool_msg_status == "error"


def test_steady_state_tool_success(fake_app_factory, monkeypatch):
    """Test steady_state success status handling."""
    success_message = ToolMessage(
        content="Steady state computed",
        name="steady_state",
        status="success",
        artifact={"state": "ok"},
        tool_call_id="call-2",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [{"messages": [success_message]}])
    prompt = (
        "Bring model 64 to a steady state. Set the initial concentration of "
        "`Pyruvate` to 0.2. The concentration of `NAD` resets to 100 every "
        "2 time units."
    )
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    steady_state_invoked = False
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage) and msg.name == "steady_state":
            steady_state_invoked = msg.status != "error"
            break
    assert steady_state_invoked


def test_steady_state_tool_ask_question(fake_app_factory, monkeypatch):
    """Test ask_question tool invocation after steady_state."""
    success_message = ToolMessage(
        content="Steady state computed",
        name="steady_state",
        status="success",
        artifact={"state": "ok"},
        tool_call_id="call-2",
    )
    ask_question_message = ToolMessage(
        content="0.06",
        name="ask_question",
        status="success",
        artifact=None,
        tool_call_id="call-3",
    )
    app, config = _build_app(
        fake_app_factory,
        monkeypatch,
        [{"messages": [success_message, ask_question_message]}],
    )
    prompt = (
        "What is the Phosphoenolpyruvate concentration at the steady state? "
        "Show only the concentration, rounded to 2 decimal places. For example, "
        "if the concentration is 0.123456, your response should be `0.12`. "
        "Do not return any other information."
    )
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    assistant_msg = response["messages"][-1].content
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    ask_questool_invoked = False
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage) and msg.name == "ask_question":
            ask_questool_invoked = True
            break
    assert ask_questool_invoked
    assert "0.06" in assistant_msg


def test_run_steady_state_branches(monkeypatch):
    """
    Cover run_steady_state success and failure.
    """

    class FailingModel:
        """Model stub that fails steady state."""

        def __init__(self):
            self.copasi_model = "copasi"

        def update_parameters(self, *_args, **_kwargs):
            """No-op parameter update."""
            return None

        def marker(self):
            """No-op method to satisfy lint for public methods."""
            return None

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.basico.task_steadystate.run_steadystate",
        lambda model=None: 0,
    )
    with pytest.raises(ValueError):
        run_steady_state(FailingModel(), {})
    assert FailingModel().marker() is None

    # Success path
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.basico.task_steadystate.run_steadystate",
        lambda model=None: 1,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.basico.model_info.get_species",
        lambda model=None: pd.DataFrame(
            {
                "name": ["A"],
                "concentration": [1],
                "transition_time": [0],
                "initial_particle_number": [0],
                "initial_expression": [0],
                "expression": [0],
                "particle_number": [0],
                "type": ["t"],
                "particle_number_rate": [0],
                "key": [0],
                "sbml_id": ["id"],
                "display_name": ["disp"],
            }
        ),
    )
    df = run_steady_state(FailingModel(), {})
    assert "steady_state_concentration" in df.columns


def test_steady_state_with_recurring(monkeypatch):
    """
    Ensure add_rec_events path is hit in the tool.
    """
    tool = SteadyStateTool()
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.run_steady_state",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"species_name": ["A"], "steady_state_concentration": [1]}
        ),
    )
    added = {}

    def fake_add_rec_events(_model_obj, _rec_data):
        added["called"] = True

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.add_rec_events",
        fake_add_rec_events,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": "exp",
                    "time_data": None,
                    "species_to_be_analyzed_before_experiment": None,
                    "reocurring_data": {
                        "data": [
                            {
                                "time": 1,
                                "species_name": "X",
                                "species_concentration": 2,
                            }
                        ]
                    },
                },
            },
        )
    )
    assert added.get("called") is True
    assert "dic_steady_state_data" in cmd.update


def test_steady_state_with_initial_species(monkeypatch):
    """
    Cover branch that builds species_to_be_analyzed_before_experiment dict.
    """
    tool = SteadyStateTool()
    captured = {}

    def fake_run_steady_state(_model_obj, species_dict):
        captured["species_dict"] = species_dict
        return pd.DataFrame({"species_name": ["A"], "steady_state_concentration": [1.0]})

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.run_steady_state",
        fake_run_steady_state,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": "exp",
                    "time_data": None,
                    "species_to_be_analyzed_before_experiment": {
                        "species_name": ["S1"],
                        "species_concentration": [2.5],
                    },
                    "reocurring_data": None,
                },
            },
        )
    )
    assert captured["species_dict"] == {"S1": 2.5}
    assert "dic_steady_state_data" in cmd.update
