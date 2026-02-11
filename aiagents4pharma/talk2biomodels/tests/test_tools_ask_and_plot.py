"""
Unit-mock coverage for ask_question and custom_plotter tools.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..tools.ask_question import AskQuestionTool
from ..tools.custom_plotter import CustomPlotterTool
from ..tools.load_biomodel import ModelData

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


def test_ask_question_tool(monkeypatch):
    """Exercise ask_question tool path with mocked dependencies."""
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
        "llm_model": MagicMock(),
        "dic_simulated_data": [{"name": "exp", "data": {"Time": [0], "X": [1]}}],
        "dic_steady_state_data": [{"name": "exp", "data": {"Time": [0], "X": [2]}}],
    }
    result = tool.invoke(
        _tool_call(
            tool,
            {
                "question": "q",
                "experiment_name": "exp",
                "question_context": "simulation",
                "state": state,
            },
        )
    )
    content = result.content if hasattr(result, "content") else result
    assert content == "answer"


def test_custom_plotter_tool(monkeypatch):
    """Exercise custom_plotter tool path with mocked dependencies."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.extract_relevant_species",
        lambda question, species_names, state: SimpleNamespace(relevant_species=["A"]),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.load_biomodel",
        lambda *args, **kwargs: SimpleNamespace(copasi_model="copasi"),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.custom_plotter.get_model_units",
        lambda model_obj=None: {"y_axis_label": "mol", "x_axis_label": "s"},
    )
    tool = CustomPlotterTool()
    state = {
        "sbml_file_path": [],
        "llm_model": MagicMock(),
        "dic_simulated_data": [{"name": "sim", "data": {"Time": [0], "A": [1], "B": [2]}}],
    }
    result = tool.invoke(
        _tool_call(
            tool,
            {
                "question": "plot A",
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                "simulation_name": "sim",
                "state": state,
            },
        )
    )
    content, artifact = result.content, result.artifact
    assert "Custom plot" in content
    assert artifact["dic_data"][0]["A"] == 1
