'''
Test cases for plot_figure.py
'''

import streamlit as st
from ..models.basico_model import BasicoModel
from ..tools.fetch_parameters import FetchParametersTool

ST_SESSION_KEY = "test_key"
MODEL_OBJ = BasicoModel(model_id=537)

def test_tool_fetch_params():
    '''
    Test the tool fetch_params.
    '''
    st.session_state[ST_SESSION_KEY] = MODEL_OBJ
    fetch_params = FetchParametersTool(st_session_key=ST_SESSION_KEY)
    response = fetch_params.invoke(input={
                                'fetch_species': True,
                                'fetch_parameters': True
                                })
    # Check if response is a dictionary
    # with keys 'Species' and 'Parameters'
    assert isinstance(response, dict)
    assert 'Species' in response
    assert 'Parameters' in response
