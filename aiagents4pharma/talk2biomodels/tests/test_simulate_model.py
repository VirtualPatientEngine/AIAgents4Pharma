'''
Test cases for simulate_model.py
'''

import pytest
import streamlit as st
from ..tools.simulate_model import (SimulateModelTool,
                                    ModelData,
                                    TimeData,
                                    SpeciesData,
                                    TimeSpeciesNameConcentration,
                                    RecurringData)

@pytest.fixture(name="simulate_model_tool")
def simulate_model_tool_fixture():
    '''
    Fixture for creating an instance of SimulateModelTool.
    '''
    return SimulateModelTool(st_session_key="test_key")

def test_run_with_valid_modelid(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a valid model ID.
    '''
    model_data=ModelData(modelid=64)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    time_species_name_concentration=TimeSpeciesNameConcentration(time=10.0,
                                                                species_name="Pyruvate",
                                                                species_concentration=10.0)
    recurring_data=RecurringData(data=[time_species_name_concentration])
    st.session_state["test_key"] = None
    result = simulate_model_tool.invoke(input={'model_data':model_data,
                                        'time_data':time_data,
                                        'species_data':species_data,
                                        'recurring_data':recurring_data})
    assert result == "Simulation results for the model."

def test_run_with_invalid_modelid(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with an invalid model ID.
    '''
    # sleep(5)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st.session_state["test_key"] = None
    result = simulate_model_tool.invoke(input={
                                          'time_data':time_data,
                                          'species_data':species_data})
    assert result == "Please provide a BioModels ID or an SBML file path for simulation."
    # slice(5)
    model_data=ModelData(modelid=64)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    simulate_model_tool.invoke(input={'model_data':model_data,
                                          'time_data':time_data,
                                          'species_data':species_data})
    result = simulate_model_tool.invoke(input={'time_data':time_data,
                                          'species_data':species_data})
    assert result == "Simulation results for the model."

def test_run_with_missing_session_key(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class 
    with a missing session key.
    '''
    model_data=ModelData(modelid=64)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    # Delete existing session key, if any
    del st.session_state["test_key"]
    result = simulate_model_tool.invoke(input={'model_data':model_data,
                                          'time_data':time_data,
                                          'species_data':species_data})
    expected_result = f"Session key {simulate_model_tool.st_session_key} "
    expected_result += "not found in Streamlit session state."
    assert result == expected_result

def test_run_with_valid_sbml_file_path(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class 
    with a valid SBML file path.
    '''
    sbml_file_path="aiagents4pharma/talk2biomodels/tests/BIOMD0000000064.xml"
    model_data=ModelData(sbml_file_path=sbml_file_path)
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st.session_state["test_key"] = None
    result = simulate_model_tool.invoke(input={'model_data':model_data,
                                          'time_data':time_data,
                                          'species_data':species_data})
    assert result == "Simulation results for the model."

def test_run_with_no_modelid_or_sbml_file_path(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class 
    with a missing model ID and SBML file path.
    '''
    time_data=TimeData(duration=100.0, interval=10)
    species_data=SpeciesData(species_name=["Pyruvate"], species_concentration=[1.0])
    st.session_state["test_key"] = None
    st.session_state["sbml_file_path"] = None
    result = simulate_model_tool.invoke(input={'time_data':time_data,
                                          'species_data':species_data})
    assert result == "Please provide a BioModels ID or an SBML file path for simulation."
    simulate_model_tool.st_session_key = None
    result = simulate_model_tool.invoke(input={'time_data':time_data,
                                          'species_data':species_data})
    assert result == "Please provide a BioModels ID or an SBML file path for simulation."

def test_get_metadata(simulate_model_tool):
    '''
    Test the get_metadata method of the SimulateModelTool class.
    '''
    metadata = simulate_model_tool.get_metadata()
    assert metadata["name"] == "simulate_model"
    assert metadata["description"] == "A tool to simulate a model."
