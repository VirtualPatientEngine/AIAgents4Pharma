"""
Additional branch coverage for ask_question, custom_plotter, get_annotation,
load_biomodel, parameter_scan, simulate_model.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from ..tools.ask_question import AskQuestionTool
from ..tools.custom_plotter import CustomPlotterTool
from ..tools.get_annotation import extract_relevant_species_names
from ..tools.load_biomodel import ModelData, load_biomodel
from ..tools.parameter_scan import make_list_dic_scanned_data, run_parameter_scan
from ..tools.simulate_model import SimulateModelTool

pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


class FakeCtx:
    """Hydra initialize context manager stub."""

    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def test_fake_ctx_methods():
    """Exercise FakeCtx enter/exit behavior."""
    with FakeCtx() as ctx:
        assert ctx is None


def test_ask_question_steady_state(monkeypatch):
    """Exercise ask_question steady_state branch."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.ask_question.hydra.initialize",
        lambda **kwargs: FakeCtx(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.ask_question.hydra.compose",
        lambda **kwargs: SimpleNamespace(
            tools=SimpleNamespace(
                ask_question=SimpleNamespace(
                    simulation_prompt="sim:",
                    steady_state_prompt="ss:",
                )
            )
        ),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.ask_question.create_pandas_dataframe_agent",
        lambda *_args, **_kwargs: SimpleNamespace(invoke=lambda *_a, **_k: {"output": "answer"}),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.ask_question.basico.model_info.get_model_units",
        lambda model=None: {"quantity_unit": "mol", "time_unit": "s"},
    )
    tool = AskQuestionTool()
    state = {
        "llm_model": SimpleNamespace(),
        "dic_simulated_data": [{"name": "exp", "data": {"Time": [0], "X": [1]}}],
        "dic_steady_state_data": [{"name": "exp", "data": {"Time": [0], "X": [2]}}],
    }
    result = tool.invoke(
        _tool_call(
            tool,
            {
                "question": "q",
                "experiment_name": "exp",
                "question_context": "steady_state",
                "state": state,
            },
        )
    )
    content = result.content if hasattr(result, "content") else result
    assert content == "answer"


def test_custom_plotter_no_species(monkeypatch):
    """Ensure custom_plotter raises when no species are extracted."""
    tool = CustomPlotterTool()
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.extract_relevant_species",
        lambda question, species_names, state: SimpleNamespace(relevant_species=None),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.load_biomodel",
        lambda *args, **kwargs: SimpleNamespace(copasi_model="copasi"),
    )
    state = {
        "sbml_file_path": [],
        "llm_model": SimpleNamespace(),
        "dic_simulated_data": [{"name": "sim", "data": {"Time": [0], "A": [1]}}],
    }
    with pytest.raises(ValueError):
        tool.invoke(
            _tool_call(
                tool,
                {
                    "question": "none",
                    "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                    "simulation_name": "sim",
                    "state": state,
                },
            )
        )


def test_extract_relevant_species_names_error(monkeypatch):
    """Cover extract_relevant_species_names error when no species match."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.hydra.initialize",
        lambda **kwargs: FakeCtx(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.hydra.compose",
        lambda **kwargs: SimpleNamespace(
            tools=SimpleNamespace(get_annotation=SimpleNamespace(prompt="prompt"))
        ),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.model_info.get_species",
        lambda model=None: pd.DataFrame(index=["S1"]),
    )
    llm = SimpleNamespace()
    llm.with_structured_output = lambda model: SimpleNamespace(
        invoke=lambda question: SimpleNamespace(relevant_species=None)
    )
    with pytest.raises(ValueError):
        extract_relevant_species_names(
            SimpleNamespace(copasi_model="copasi"),
            SimpleNamespace(user_question="uq"),
            {"llm_model": llm},
        )


def test_load_biomodel_returns_none():
    """Ensure load_biomodel returns None when no inputs are provided."""
    assert load_biomodel(SimpleNamespace(biomodel_id=None), sbml_file_path=None) is None


def test_run_parameter_scan_invalid(monkeypatch):
    """Cover run_parameter_scan invalid parameter branch."""
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
        """Minimal model stub for invalid parameter branch."""

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
            species_parameter_name="Invalid",
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


def test_make_list_dic_scanned_data():
    """Cover make_list_dic_scanned_data helper."""
    dic_param_scan = {"S1": pd.DataFrame({"Time": [0], "S1_1": [1]})}
    arg_data = SimpleNamespace(experiment_name="exp")
    sys_bio_model = ModelData(biomodel_id="BIOMD1")
    result = make_list_dic_scanned_data(dic_param_scan, arg_data, sys_bio_model, "tc")
    assert result[0]["name"] == "exp:S1"


def test_simulate_model_with_time_and_recurring(monkeypatch):
    """Cover simulate_model with time and recurring data."""
    dummy_model = SimpleNamespace(
        copasi_model="copasi",
        biomodel_id="BIOMD1",
        update_parameters=lambda params: None,
        simulate=lambda duration, interval: pd.DataFrame({"Time": [0], "X": [1]}),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.simulate_model.add_rec_events",
        lambda model_obj, rec: setattr(model_obj, "rec_added", True),
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
    arg_data = SimpleNamespace(
        experiment_name="exp",
        species_to_be_analyzed_before_experiment=SimpleNamespace(
            species_name=["X"], species_concentration=[1]
        ),
        reocurring_data=SimpleNamespace(
            data=[SimpleNamespace(time=1, species_name="X", species_concentration=2)]
        ),
        time_data=SimpleNamespace(duration=5, interval=1),
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
                    "reocurring_data": {
                        "data": [
                            {
                                "time": arg_data.reocurring_data.data[0].time,
                                "species_name": (arg_data.reocurring_data.data[0].species_name),
                                "species_concentration": (
                                    arg_data.reocurring_data.data[0].species_concentration
                                ),
                            }
                        ]
                    },
                },
            },
        )
    )
    assert cmd.update["dic_simulated_data"][0]["source"] == "BIOMD1"
