"""
Unit-mock coverage for parameter_scan, simulate_model, and steady_state tools.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from ..tools.load_biomodel import ModelData
from ..tools.parameter_scan import ParameterScanTool
from ..tools.simulate_model import SimulateModelTool
from ..tools.steady_state import SteadyStateTool

pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def test_parameter_scan_tool(monkeypatch):
    """Exercise parameter_scan tool with mocked model and units."""

    class DummyModel:
        """Minimal model with parameter updates and simulation results."""

        def __init__(self):
            self.copasi_model = "copasi"
            self.simulation_results = pd.DataFrame({"Time": [0, 1], "S1": [1, 1]})
            self.params = None

        def model_copy(self):
            """Return a new model instance."""
            return DummyModel()

        def update_parameters(self, params):
            """Capture parameters for assertions."""
            self.params = params

        def simulate(self, *_args, **_kwargs):
            """Return simulated data."""
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
        time_data=None,
        species_to_be_analyzed_before_experiment=None,
        parameter_scan_data=SimpleNamespace(
            species_names=["S1"], species_parameter_name="S1", species_parameter_values=[1, 2]
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


def test_simulate_model_tool(monkeypatch):
    """Exercise simulate_model tool with mocked model and units."""
    dummy_model = SimpleNamespace(
        copasi_model="copasi",
        biomodel_id="BIOMD1",
        update_parameters=lambda params: None,
        simulate=lambda duration, interval: pd.DataFrame({"Time": [0], "X": [1]}),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.simulate_model.load_biomodel",
        lambda *args, **kwargs: dummy_model,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.simulate_model.basico.model_io.save_model_to_string",
        lambda: "<xml/>",
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.simulate_model.get_model_units",
        lambda model_obj=None: {"y_axis_label": "mol", "x_axis_label": "s"},
    )
    tool = SimulateModelTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": "exp",
                    "time_data": {"duration": 1, "interval": 1},
                    "species_to_be_analyzed_before_experiment": None,
                    "reocurring_data": None,
                },
            },
        )
    )
    assert "dic_simulated_data" in cmd.update


def test_steady_state_tool(monkeypatch):
    """Exercise steady_state tool with mocked model and steady state."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.run_steady_state",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"species_name": ["A"], "steady_state_concentration": [1]}
        ),
    )
    tool = SteadyStateTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": "exp",
                    "time_data": {"duration": 1, "interval": 1},
                    "species_to_be_analyzed_before_experiment": None,
                    "reocurring_data": None,
                },
            },
        )
    )
    assert "dic_steady_state_data" in cmd.update


def test_steady_state_tool_no_argdata(monkeypatch):
    """
    Cover steady_state _run when arg_data is None.
    """
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.steady_state.run_steady_state",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"species_name": ["A"], "steady_state_concentration": [1]}
        ),
    )
    tool = SteadyStateTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "arg_data": {
                    "experiment_name": "exp",
                    "time_data": {"duration": 1, "interval": 1},
                    "species_to_be_analyzed_before_experiment": None,
                    "reocurring_data": None,
                },
            },
        )
    )
    assert "dic_steady_state_data" in cmd.update
