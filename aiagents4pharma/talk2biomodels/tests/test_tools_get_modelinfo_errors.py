"""
Cover error branches in get_modelinfo tool.
"""

import pandas as pd
import pytest

from ..tools.get_modelinfo import GetModelInfoTool
from ..tools.load_biomodel import ModelData

pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def test_get_modelinfo_errors(monkeypatch):
    """Cover get_modelinfo error branches."""
    tool = GetModelInfoTool()
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.load_biomodel",
        lambda *_args, **_kwargs: type(
            "Dummy",
            (),
            {
                "copasi_model": "copasi",
                "biomodel_id": "BIOMD1",
                "description": "d",
                "name": "n",
            },
        )(),
    )
    # Species missing
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_species",
        lambda model=None: None,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_parameters",
        lambda model=None: pd.DataFrame(),
    )
    with pytest.raises(ValueError):
        tool.invoke(
            _tool_call(
                tool,
                {
                    "requested_model_info": {"species": True},
                    "tool_call_id": "tc",
                    "state": {"sbml_file_path": []},
                    "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                },
            )
        )

    # Parameters missing
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_species",
        lambda model=None: pd.DataFrame(
            {
                "name": ["s1"],
                "compartment": ["c"],
                "type": ["t"],
                "unit": ["u"],
                "initial_concentration": [1],
                "display_name": ["d"],
            }
        ),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_modelinfo.basico.model_info.get_parameters",
        lambda model=None: None,
    )
    with pytest.raises(ValueError):
        tool.invoke(
            _tool_call(
                tool,
                {
                    "requested_model_info": {"parameters": True},
                    "tool_call_id": "tc",
                    "state": {"sbml_file_path": []},
                    "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
                },
            )
        )
