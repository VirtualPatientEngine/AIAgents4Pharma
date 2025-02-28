'''
Test cases for Talk2Biomodels.
'''

import pytest
from ..tools.load_biomodel import ModelData

def test_model_data_valid_biomodel_id():
    '''
    Test the ModelData class with valid
    biomodel
    '''
    # Test with string biomodel_id starting with 'BIOMD'
    model_data = ModelData(biomodel_id='BIOMD0000000537')
    assert model_data.biomodel_id == 'BIOMD0000000537'

    # Test with string biomodel_id starting with 'MODEL'
    model_data = ModelData(biomodel_id='MODEL0000000537')
    assert model_data.biomodel_id == 'MODEL0000000537'

def test_model_data_invalid_biomodel_id():
    '''
    Test the ModelData class with invalid
    biomodel
    '''
    # Test with invalid string biomodel_id
    with pytest.raises(ValueError):
        ModelData(biomodel_id='12345')

    # Test with float biomodel_id
    with pytest.raises(ValueError):
        ModelData(biomodel_id=123.45)
