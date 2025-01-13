'''
Test cases for plot_figure.py
'''

import pytest
import streamlit as st
from ..tools.custom_plotter import CustomPlotterTool
from ..models.basico_model import BasicoModel

ST_SESSION_KEY = "test_key"
QUESTION = "Strictly plot only T-helper cells related species. Do not plot any other species."

@pytest.fixture(name="custom_plotter_tool")
def custom_plotter_tool_fixture():
    '''
    Fixture for creating an instance of custom_plotter_tool.
    '''
    return CustomPlotterTool(st_session_key=ST_SESSION_KEY)

def test_tool(custom_plotter_tool):
    '''
    Test the tool custom_plotter_tool.
    '''
    custom_plotter = custom_plotter_tool
    st.session_state[ST_SESSION_KEY] = None
    response = custom_plotter.invoke(input={
                    'question': QUESTION
                    })
    assert response == "Please run the simulation first before plotting the figure."
    st.session_state[ST_SESSION_KEY] = BasicoModel(model_id=537)
    response = custom_plotter.invoke(input={
                    'question': QUESTION
                    })
    assert response == "Please run the simulation first before plotting the figure."
    st.session_state[ST_SESSION_KEY].simulate()
    response = custom_plotter.invoke(input={
                    'question': "Plot only `IL100` species"
                    })
    assert response.startswith("No species found in the simulation")
    response = custom_plotter.invoke(input={
                    'question': 'Plot only antibodies'
                    })
    assert response.startswith("Plotted the figure")
