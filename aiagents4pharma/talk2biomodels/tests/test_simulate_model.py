'''
Test cases for simulate_model.py
'''

from time import sleep
import pytest
import streamlit as st
from ..tools.simulate_model import (SimulateModelTool,
                                    ModelData,
                                    TimeData,
                                    SpeciesData)

@pytest.fixture(name="simulate_model_tool")
def simulate_model_tool_fixture():
    '''
    Fixture for creating an instance of SimulateModelTool.
    '''
    return SimulateModelTool()

def test_run_with_valid_modelid(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a valid model ID.
    '''
    model_data=ModelData(modelid=64)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st_session_key="test_key"
    st.session_state["test_key"] = None
    result = simulate_model_tool.call_run(model_data=model_data,
                                          time_data=time_data,
                                          species_data=species_data,
                                          st_session_key=st_session_key)
    assert result == f"Simulation results for the model {model_data.modelid}."

def test_run_with_invalid_modelid(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with an invalid model ID.
    '''
    sleep(5)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st_session_key="test_key"
    st.session_state["test_key"] = None
    result = simulate_model_tool.call_run(model_data=None,
                                        time_data=time_data,
                                        species_data=species_data,
                                        st_session_key=st_session_key)
    assert result == "Please provide a BioModels ID or an SBML file path for simulation."
    slice(5)
    model_data=ModelData(modelid=64)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st_session_key="test_key"
    simulate_model_tool.call_run(model_data=model_data,
                                time_data=time_data,
                                species_data=species_data,
                                st_session_key=st_session_key)
    result = simulate_model_tool.call_run(model_data=None,
                                        time_data=time_data,
                                        species_data=species_data,
                                        st_session_key=st_session_key)
    assert result == "Simulation results for the model 64."

def test_run_with_missing_session_key(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a missing session key.
    '''
    model_data=ModelData(modelid=64)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st_session_key="new_test_key"
    result = simulate_model_tool.call_run(model_data=model_data,
                                        time_data=time_data,
                                        species_data=species_data,
                                        st_session_key=st_session_key)
    assert result == f"Session key {st_session_key} not found in Streamlit session state."

def test_run_with_valid_sbml_file_path(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a valid SBML file path.
    '''
    sbml_file_path="./BIOMD0000000064.xml"
    model_data=ModelData(sbml_file_path=sbml_file_path)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st_session_key="test_key"
    st.session_state["test_key"] = None

    result = simulate_model_tool.call_run(model_data=model_data,
                                          time_data=time_data,
                                          species_data=species_data,
                                          st_session_key=st_session_key)
    assert result == "Simulation results for the model internal."
    st.session_state["sbml_file_path"] = model_data.sbml_file_path
    result = simulate_model_tool.call_run(time_data=time_data,
                                          species_data=species_data,
                                          st_session_key=st_session_key)
    assert result == "Simulation results for the model internal."

def test_run_with_no_modelid_or_sbml_file_path(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a missing model ID and SBML file path.
    '''
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st_session_key="test_key"
    st.session_state["test_key"] = None
    st.session_state["sbml_file_path"] = None
    result = simulate_model_tool.call_run(time_data=time_data,
                                          species_data=species_data,
                                          st_session_key=st_session_key)
    assert result == "Please provide a BioModels ID or an SBML file path for simulation."
    result = simulate_model_tool.call_run(time_data=time_data,
                                          species_data=species_data,
                                          st_session_key=None)
    assert result == "Please provide a BioModels ID or an SBML file path for simulation."

def test_get_metadata(simulate_model_tool):
    '''
    Test the get_metadata method of the SimulateModelTool class.
    '''
    metadata = simulate_model_tool.get_metadata()
    assert metadata["name"] == "simulate_model"
    assert metadata["description"] == "A tool for simulating a model."
    assert metadata["return_direct"] == simulate_model_tool.return_direct
