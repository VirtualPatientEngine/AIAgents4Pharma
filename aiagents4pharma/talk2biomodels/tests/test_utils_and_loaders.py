"""
Unit-mock coverage for utils and loader helpers.
"""

from types import SimpleNamespace

import pytest

from ..tools import utils
from ..tools.load_arguments import ReocurringData, TimeSpeciesNameConcentration, add_rec_events
from ..tools.load_biomodel import ModelData, ensure_biomodel_id, load_biomodel

pytestmark = pytest.mark.unit_mock


def test_utils_and_load_biomodel(monkeypatch):
    """Exercise utils and load_biomodel helpers."""
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.utils.basico.model_info.get_model_units",
        lambda model=None: {"quantity_unit": "mol", "time_unit": "s"},
    )
    dummy_model = SimpleNamespace(copasi_model="copasi")
    assert utils.get_model_units(dummy_model) == {"y_axis_label": "mol", "x_axis_label": "s"}

    with pytest.raises(ValueError):
        ensure_biomodel_id("bad")
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.load_biomodel.BasicoModel",
        lambda biomodel_id=None, sbml_file_path=None: SimpleNamespace(
            biomodel_id=biomodel_id, sbml_file_path=sbml_file_path
        ),
    )
    assert ensure_biomodel_id(5) == 5
    assert load_biomodel(ModelData(biomodel_id="BIOMD1"), None).biomodel_id == "BIOMD1"
    assert load_biomodel(SimpleNamespace(biomodel_id=None), "file.xml").sbml_file_path == "file.xml"


def test_add_rec_events(monkeypatch):
    """Cover add_rec_events wiring to basico.add_event."""
    captured = []

    def fake_add_event(name, *_args, **_kwargs):
        captured.append(name)

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.load_arguments.basico.add_event",
        fake_add_event,
    )
    rec = ReocurringData(
        data=[TimeSpeciesNameConcentration(time=1, species_name="S", species_concentration=2)]
    )
    add_rec_events(SimpleNamespace(copasi_model="copasi"), rec)
    assert captured == ["S_1"]
