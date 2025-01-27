'''
This file contains the unit tests for the BioModel class.
'''

from typing import List, Dict, Union, Optional
from pydantic import Field
import pytest
from ..models.sys_bio_model import SysBioModel

class TestBioModel(SysBioModel):
    '''
    A test BioModel class for unit testing.
    '''

    biomodel_id: Optional[int] = Field(None, description="BioModel ID of the model")
    sbml_file_path: Optional[str] = Field(None, description="Path to an SBML file")
    name: Optional[str] = Field(..., description="Name of the model")
    description: Optional[str] = Field("", description="Description of the model")
    param1: Optional[float] = Field(0.0, description="Parameter 1")
    param2: Optional[float] = Field(0.0, description="Parameter 2")

    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        '''
        Get the metadata of the model.
        '''
        return self.biomodel_id

    def update_parameters(self, parameters):
        '''
        Update the model parameters.
        '''
        self.param1 = parameters.get('param1', 0.0)
        self.param2 = parameters.get('param2', 0.0)

    def simulate(self, duration: Union[int, float]) -> List[float]:
        '''
        Simulate the model.
        '''
        return [self.param1 + self.param2 * t for t in range(int(duration))]

def test_get_model_metadata():
    '''
    Test the get_model_metadata method of the BioModel class.
    '''
    model = TestBioModel(biomodel_id=123, name="Test Model", description="A test model")
    metadata = model.get_model_metadata()
    assert metadata == 123

def test_check_biomodel_id_or_sbml_file_path():
    '''
    Test the check_biomodel_id_or_sbml_file_path method of the BioModel class.
    '''
    with pytest.raises(ValueError):
        TestBioModel(name="Test Model", description="A test model")

def test_simulate():
    '''
    Test the simulate method of the BioModel class.
    '''
    model = TestBioModel(biomodel_id=123, name="Test Model", description="A test model")
    model.update_parameters({'param1': 1.0, 'param2': 2.0})
    results = model.simulate(duration=4.0)
    assert results == [1.0, 3.0, 5.0, 7.0]
