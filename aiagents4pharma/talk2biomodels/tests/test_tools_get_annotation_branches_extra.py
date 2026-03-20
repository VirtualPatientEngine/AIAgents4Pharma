"""
Cover remaining branches in get_annotation processing via tool.invoke.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from ..tools.get_annotation import GetAnnotationTool
from ..tools.load_biomodel import ModelData

pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def test_fetch_descriptions_other_db(monkeypatch):
    """
    Ensure 'other' database branch populates results via tool.invoke.
    """
    tool = GetAnnotationTool()
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.extract_relevant_species_names",
        lambda *_args, **_kwargs: ["S1"],
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.search_uniprot_labels",
        lambda ids: {},
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.search_ols_labels",
        lambda data: {},
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.get_miriam_annotation",
        lambda name: {
            "descriptions": [
                {"id": "http://db/otherdb/X1", "qualifier": "is"},
                {"id": "http://db/otherdb/NaN", "qualifier": "is"},
            ]
        },
    )
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "arg_data": {"experiment_name": "exp", "user_question": "q"},
                "tool_call_id": "tc",
                "state": {"sbml_file_path": [], "llm_model": SimpleNamespace()},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
            },
        )
    )
    df = pd.DataFrame(cmd.update["dic_annotations_data"][0]["data"])
    assert "-" in df["Description"].values
