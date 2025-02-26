'''
A test BasicoModel class for pytest unit testing.
'''

import pandas as pd
import pytest
import basico
from ..models.basico_model import BasicoModel

@pytest.fixture(name="model")
def model_fixture():
    """
    A fixture for the BasicoModel class.
    """
    return BasicoModel(biomodel_id=64, species={"Pyruvate": 100}, duration=2, interval=2)

def test_with_biomodel_id(model):
    """
    Test initialization of BasicoModel with biomodel_id.
    """
    assert model.biomodel_id == 64
    model.update_parameters(parameters={'Pyruvate': 0.5, 'KmPFKF6P': 1.5})
    df_species = basico.model_info.get_species(model=model.copasi_model)
    assert df_species.loc['Pyruvate', 'initial_concentration'] == 0.5
    df_parameters = basico.model_info.get_parameters(model=model.copasi_model)
    assert df_parameters.loc['KmPFKF6P', 'initial_value'] == 1.5
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
        model.update_parameters(parameters={'Pyruv': 0.5})

def test_with_sbml_file():
    """
    Test initialization of BasicoModel with sbml_file_path.
    """
    model_object = BasicoModel(sbml_file_path="./BIOMD0000000064_url.xml")
    assert model_object.sbml_file_path == "./BIOMD0000000064_url.xml"
    assert isinstance(model_object.simulate(duration=2, interval=2), pd.DataFrame)

def test_check_biomodel_id_or_sbml_file_path():
    '''
    Test the check_biomodel_id_or_sbml_file_path method of the BioModel class.
    '''
    with pytest.raises(ValueError):
        BasicoModel(species={"Pyruvate": 100}, duration=2, interval=2)

def test_get_model_metadata():
    """
    Test the get_model_metadata method of the BasicoModel class.
    """
    model = BasicoModel(biomodel_id=64)
    metadata = model.get_model_metadata()
    assert metadata["Model Type"] == "SBML Model (COPASI)"
    assert metadata["Parameter Count"] == len(basico.get_parameters())
