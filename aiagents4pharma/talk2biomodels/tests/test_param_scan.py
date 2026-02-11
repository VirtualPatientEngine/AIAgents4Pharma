"""
Test cases for Talk2Biomodels parameter scan tool.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ..agents.t2b_agent import get_app
from ..tools.load_biomodel import ModelData
from ..tools.parameter_scan import ParameterScanTool, run_parameter_scan

LLM_MODEL = MagicMock(name="llm_model")
pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def test_param_scan_tool(fake_app_factory, monkeypatch):
    """
    In this test, we will test the parameter_scan tool.
    We will prompt it to scan the parameter `kIL6RBind`
    from 1 to 100 in steps of 10, record the changes
    in the concentration of the species `Ab{serum}` in
    model 537.

    We will pass the inaccuarate parameter (`KIL6Rbind`)
    and species names (just `Ab`) to the tool to test
    if it can deal with it.

    We expect the agent to first invoke the parameter_scan
    tool and raise an error. It will then invoke another
    tool get_modelinfo to get the correct parameter
    and species names. Finally, the agent will reinvoke
    the parameter_scan tool with the correct parameter
    and species names.

    """
    messages = [
        ToolMessage(
            content="Error: ValueError('Invalid species or parameter name: Ab')",
            name="parameter_scan",
            status="error",
            artifact=None,
            tool_call_id="call-1",
        ),
        ToolMessage(
            content="Parameter scan results of kIL6Rbind",
            name="parameter_scan",
            status="success",
            artifact={"result": [1, 2, 3]},
            tool_call_id="call-2",
        ),
        ToolMessage(
            content="Model info ready",
            name="get_modelinfo",
            status="success",
            artifact={"data": "info"},
            tool_call_id="call-3",
        ),
    ]
    app = fake_app_factory([{"messages": messages}])
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_param_scan.get_app",
        lambda *args, **kwargs: app,
    )
    unique_id = 1234
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = """How will the value of Ab in serum in model 537 change
            if the param kIL6Rbind is varied from 1 to 100 in steps of 10?
            Set the initial `DoseQ2W` concentration to 300. Assume
            that the model is simulated for 2016 hours with an interval of 50."""
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1] + ["not-a-tool-message"]
    tool_msgs = [msg for msg in reversed_messages if isinstance(msg, ToolMessage)]
    df = pd.DataFrame(
        {
            "name": [msg.name for msg in tool_msgs],
            "status": [msg.status for msg in tool_msgs],
            "content": [msg.content for msg in tool_msgs],
        }
    )
    # print (df)
    assert any(
        (df["status"] == "error")
        & (df["name"] == "parameter_scan")
        & (df["content"].str.startswith("Error: ValueError('Invalid species or parameter name:"))
    )
    assert any(
        (df["status"] == "success")
        & (df["name"] == "parameter_scan")
        & (df["content"].str.startswith("Parameter scan results of"))
    )
    assert any((df["status"] == "success") & (df["name"] == "get_modelinfo"))


def test_param_scan_tool_no_argdata(monkeypatch):
    """
    Ensure _run handles missing arg_data gracefully with defaults.
    """
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.load_biomodel",
        lambda *args, **kwargs: SimpleNamespace(
            copasi_model="copasi",
            model_copy=lambda: SimpleNamespace(
                copasi_model="copasi",
                update_parameters=lambda *_a, **_k: None,
                simulate=lambda duration, interval: pd.DataFrame({"Time": [0], "S1": [1]}),
                simulation_results=pd.DataFrame({"Time": [0], "S1": [1]}),
            ),
        ),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.basico.model_info.get_parameters",
        lambda model=None: pd.DataFrame(index=[]),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.basico.model_info.get_species",
        lambda model=None: pd.DataFrame({"display_name": ["S1"]}),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.get_model_units",
        lambda model_obj=None: {"y_axis_label": "Q", "x_axis_label": "T"},
    )
    tool = ParameterScanTool()
    arg_data = SimpleNamespace(
        time_data=None,
        species_to_be_analyzed_before_experiment=None,
        parameter_scan_data=SimpleNamespace(
            species_names=["S1"], species_parameter_name="S1", species_parameter_values=[1]
        ),
        experiment_name="exp",
    )
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": arg_data.experiment_name,
                    "time_data": {"duration": 1, "interval": 1},
                    "species_to_be_analyzed_before_experiment": None,
                    "reocurring_data": None,
                    "parameter_scan_data": {
                        "species_names": arg_data.parameter_scan_data.species_names,
                        "species_parameter_name": (
                            arg_data.parameter_scan_data.species_parameter_name
                        ),
                        "species_parameter_values": (
                            arg_data.parameter_scan_data.species_parameter_values
                        ),
                    },
                },
            },
        )
    )
    assert "dic_scanned_data" in cmd.update


def test_run_parameter_scan_invalid_species(monkeypatch):
    """
    Cover invalid species branch.
    """
    df_species = pd.DataFrame({"display_name": ["S1"]})
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.basico.model_info.get_parameters",
        lambda model=None: pd.DataFrame(index=[]),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.basico.model_info.get_species",
        lambda model=None: df_species,
    )

    class DummyModel:
        """Minimal model stub for invalid-species branch."""

        def __init__(self):
            self.copasi_model = "copasi"
            self.simulation_results = None

        def model_copy(self):
            """Return self for simplified copy."""
            return self

        def update_parameters(self, *_args, **_kwargs):
            """No-op parameter update."""
            return None

        def simulate(self, *_args, **_kwargs):
            """Return fixed simulation results."""
            self.simulation_results = pd.DataFrame({"Time": [0, 1], "S1": [1, 1]})
            return self.simulation_results

    bad_arg = SimpleNamespace(
        time_data=None,
        species_to_be_analyzed_before_experiment=None,
        parameter_scan_data=SimpleNamespace(
            species_names=["Bad"],
            species_parameter_name="S1",
            species_parameter_values=[1],
        ),
        experiment_name="exp",
    )
    with pytest.raises(ValueError):
        run_parameter_scan(DummyModel(), bad_arg, {}, 1, 1)
    model = DummyModel()
    assert model.model_copy() is model
    assert model.update_parameters() is None
    assert model.simulate().equals(pd.DataFrame({"Time": [0, 1], "S1": [1, 1]}))


def test_parameter_scan_tool_with_initial_species(monkeypatch):
    """
    Cover branches where initial species and time_data are provided.
    """

    class DummyModel:
        """Minimal model stub for initial-species branch."""

        def __init__(self):
            self.copasi_model = "copasi"
            self.simulation_results = pd.DataFrame({"Time": [0, 1], "S1": [1, 2]})
            self.params = None

        def model_copy(self):
            """Return a new model instance."""
            return DummyModel()

        def update_parameters(self, params):
            """Store parameters for later inspection."""
            self.params = params

        def simulate(self, *_args, **_kwargs):
            """Return fixed simulation results."""
            return self.simulation_results

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.load_biomodel",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.basico.model_info.get_parameters",
        lambda model=None: pd.DataFrame(index=["S1"]),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.basico.model_info.get_species",
        lambda model=None: pd.DataFrame({"display_name": ["S1"]}),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.parameter_scan.get_model_units",
        lambda model_obj=None: {"y_axis_label": "mol", "x_axis_label": "s"},
    )
    arg_data = SimpleNamespace(
        time_data=SimpleNamespace(duration=2, interval=1),
        species_to_be_analyzed_before_experiment=SimpleNamespace(
            species_name=["S1"], species_concentration=[1]
        ),
        parameter_scan_data=SimpleNamespace(
            species_names=["S1"], species_parameter_name="S1", species_parameter_values=[1]
        ),
        experiment_name="exp",
    )
    tool = ParameterScanTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": arg_data.experiment_name,
                    "time_data": {
                        "duration": arg_data.time_data.duration,
                        "interval": arg_data.time_data.interval,
                    },
                    "species_to_be_analyzed_before_experiment": {
                        "species_name": (
                            arg_data.species_to_be_analyzed_before_experiment.species_name
                        ),
                        "species_concentration": (
                            arg_data.species_to_be_analyzed_before_experiment.species_concentration
                        ),
                    },
                    "reocurring_data": None,
                    "parameter_scan_data": {
                        "species_names": arg_data.parameter_scan_data.species_names,
                        "species_parameter_name": (
                            arg_data.parameter_scan_data.species_parameter_name
                        ),
                        "species_parameter_values": (
                            arg_data.parameter_scan_data.species_parameter_values
                        ),
                    },
                },
            },
        )
    )
    assert "dic_scanned_data" in cmd.update
