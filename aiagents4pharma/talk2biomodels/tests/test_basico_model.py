"""
A test BasicoModel class for pytest unit testing.
"""

import basico
import pandas as pd
import pytest

from ..models.basico_model import BasicoModel

pytestmark = pytest.mark.unit_mock


@pytest.fixture(name="mock_basico")
def mock_basico_fixture(monkeypatch):
    """
    Mock basico calls to avoid network or file access.
    """

    df_params = pd.DataFrame({"initial_value": [1.0]}, index=["KmPFKF6P"])
    df_species = pd.DataFrame({"initial_concentration": [100.0]}, index=["Pyruvate"])

    monkeypatch.setattr(basico, "load_biomodel", lambda biomodel_id: {"id": biomodel_id})
    monkeypatch.setattr(basico, "load_model", lambda path: {"path": path})
    monkeypatch.setattr(basico.biomodels, "get_model_info", lambda _id: {"description": "desc"})
    monkeypatch.setattr(basico.model_info, "get_model_name", lambda model=None: "Test Model")
    monkeypatch.setattr(basico.model_info, "get_notes", lambda model=None: "notes")
    monkeypatch.setattr(basico.model_info, "get_parameters", lambda model=None: df_params)
    monkeypatch.setattr(basico.model_info, "get_species", lambda model=None: df_species)
    monkeypatch.setattr(
        basico.model_info,
        "set_parameters",
        lambda name, initial_value, exact, model=None: df_params.__setitem__(
            "initial_value", [initial_value]
        ),
    )
    monkeypatch.setattr(
        basico.model_info,
        "set_species",
        lambda name, initial_concentration, exact, model=None: df_species.__setitem__(
            "initial_concentration", [initial_concentration]
        ),
    )
    monkeypatch.setattr(basico, "get_parameters", lambda: ["a", "b"])
    monkeypatch.setattr(
        basico,
        "run_time_course",
        lambda model=None, intervals=2, duration=2: pd.DataFrame(
            {"Time": [0, duration], "Pyruvate": [0.5, 0.5]}
        ),
    )


@pytest.fixture(name="model")
def model_fixture(mock_basico):
    """
    A fixture for the BasicoModel class.
    """
    assert mock_basico is None
    return BasicoModel(biomodel_id=64, species={"Pyruvate": 100}, duration=2, interval=2)


def test_with_biomodel_id(model):
    """
    Test initialization of BasicoModel with biomodel_id.
    """
    assert model.biomodel_id == 64
    model.update_parameters(parameters={"Pyruvate": 0.5, "KmPFKF6P": 1.5})
    df_species = basico.model_info.get_species(model=model.copasi_model)
    assert df_species.loc["Pyruvate", "initial_concentration"] == 0.5
    df_parameters = basico.model_info.get_parameters(model=model.copasi_model)
    assert df_parameters.loc["KmPFKF6P", "initial_value"] == 1.5
    # check if the simulation results are a pandas DataFrame object
    assert isinstance(model.simulate(duration=2, interval=2), pd.DataFrame)
    # Pass a None value to the update_parameters method
    # and it should not raise an error
    model.update_parameters(parameters={None: None})
    # check if the model description is updated
    assert model.description == basico.biomodels.get_model_info(model.biomodel_id)["description"]
    # check if an error is raised if an invalid species/parameter (`Pyruv`)
    # is passed and it should raise a ValueError
    with pytest.raises(ValueError):
        model.update_parameters(parameters={"Pyruv": 0.5})


def test_with_sbml_file(mock_basico):
    """
    Test initialization of BasicoModel with sbml_file_path.
    """
    assert mock_basico is None
    model_object = BasicoModel(sbml_file_path="./BIOMD0000000064_url.xml")
    assert model_object.sbml_file_path == "./BIOMD0000000064_url.xml"
    assert isinstance(model_object.simulate(duration=2, interval=2), pd.DataFrame)


def test_check_biomodel_id_or_sbml_file_path(mock_basico):
    """
    Test the check_biomodel_id_or_sbml_file_path method of the BioModel class.
    """
    assert mock_basico is None
    with pytest.raises(ValueError):
        BasicoModel(species={"Pyruvate": 100}, duration=2, interval=2)


def test_get_model_metadata(mock_basico):
    """
    Test the get_model_metadata method of the BasicoModel class.
    """
    assert mock_basico is None
    model = BasicoModel(biomodel_id=64)
    metadata = model.get_model_metadata()
    assert metadata["Model Type"] == "SBML Model (COPASI)"
    assert metadata["Parameter Count"] == len(basico.get_parameters())
