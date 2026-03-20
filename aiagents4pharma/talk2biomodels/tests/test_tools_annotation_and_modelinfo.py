"""
Unit-mock coverage for get_annotation and get_modelinfo tools.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ..tools.get_annotation import GetAnnotationTool
from ..tools.get_modelinfo import GetModelInfoTool
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


def test_get_annotation_tool(monkeypatch):
    """Exercise get_annotation tool with mocked dependencies."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.hydra.initialize",
        lambda **kwargs: FakeCtx(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.hydra.compose",
        lambda **kwargs: SimpleNamespace(
            tools=SimpleNamespace(get_annotation=SimpleNamespace(prompt="prompt: "))
        ),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    species_df = pd.DataFrame({"display_name": ["S1"]}, index=["S1"])
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.model_info.get_species",
        lambda model=None: species_df,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.get_miriam_annotation",
        lambda name: {"descriptions": [{"id": "http://uniprot.org/uniprot/P1", "qualifier": "is"}]},
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.search_uniprot_labels",
        lambda ids: {"P1": "protein"},
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.search_ols_labels",
        lambda data: {"go": {"G1": "go-label"}},
    )
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = SimpleNamespace(
        relevant_species=["S1"]
    )
    state = {"sbml_file_path": ["file.xml"], "llm_model": llm}
    tool = GetAnnotationTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "arg_data": {"experiment_name": "exp", "user_question": "S1"},
                "tool_call_id": "tc",
                "state": state,
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
            },
        )
    )
    assert "dic_annotations_data" in cmd.update


def test_get_modelinfo_tool(monkeypatch):
    """Exercise get_modelinfo tool with mocked dependencies."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(
            copasi_model="copasi", biomodel_id="BIOMD1", description="desc", name="name"
        ),
    )
    df_species = pd.DataFrame(
        {
            "name": ["s1"],
            "compartment": ["c"],
            "type": ["f"],
            "unit": ["u"],
            "initial_concentration": [1],
            "display_name": ["d"],
        }
    )
    df_params = pd.DataFrame(
        {
            "name": ["p1"],
            "type": ["t"],
            "unit": ["u"],
            "initial_value": [1],
            "display_name": ["d"],
        }
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_species",
        lambda model=None: df_species,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_parameters",
        lambda model=None: df_params,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_compartments",
        lambda model=None: pd.DataFrame(index=["c1"]),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_model_units",
        lambda model=None: {"units": "u"},
    )
    tool = GetModelInfoTool()
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "requested_model_info": {
                    "species": True,
                    "parameters": True,
                    "compartments": True,
                    "units": True,
                    "description": True,
                    "name": True,
                },
                "tool_call_id": "tc",
                "state": {"sbml_file_path": []},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
            },
        )
    )
    assert "Species" in cmd.update["messages"][0].content
