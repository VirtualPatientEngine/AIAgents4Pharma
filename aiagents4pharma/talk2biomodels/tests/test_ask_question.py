'''
Test cases for ask_question.py
'''

import pytest
import streamlit as st
from ..tools.ask_question import AskQuestionTool, AskQuestionInput, ModelData
from ..models.basico_model import BasicoModel

@pytest.fixture(name="ask_question_tool")
def ask_question_tool_fixture():
    '''
    Fixture for creating an instance of AskQuestionTool.
    '''
    return AskQuestionTool(st_session_key="test_key",
        sys_bio_model=ModelData(
            sbml_file_path="aiagents4pharma/talk2biomodels/tests//BIOMD0000000064_url.xml"
            )
            )

@pytest.fixture(name="ask_question_tool_with_model_id")
def ask_question_tool__with_model_id_fixture():
    '''
    Fixture for creating an instance of AskQuestionTool.
    '''
    return AskQuestionTool(st_session_key="test_key",
                           sys_bio_model=ModelData(modelid=64))

@pytest.fixture(name="input_data", scope="module")
def input_data_fixture():
    '''
    Fixture for creating an instance of AskQuestionInput.
    '''
    return AskQuestionInput(question="What is the concentration of Pyruvate at time 5?")

def test_run_with_sbml_file(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class 
    with a valid session key and model data.
    '''
    result = ask_question_tool.invoke(input={'question':input_data.question})
    assert isinstance(result, str)

def test_run_manager(input_data, ask_question_tool_with_model_id):
    '''
    Test the run manager of the AskQuestionTool class.
    '''
    ask_question_tool_with_model_id.metadata = {
        "prompt": "Answer the question carefully."
    }
    result = ask_question_tool_with_model_id.invoke(input={'question':input_data.question})
    assert isinstance(result, str)

def test_run_with_no_model_data_at_all(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class 
    with a valid session key and NO model data.
    '''
    ask_question_tool.sys_bio_model = ModelData()
    result = ask_question_tool.invoke(input={'question':input_data.question})
    assert isinstance(result, str)

def test_run_with_session_key(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class 
    with a missing session key.
    '''
    del st.session_state["test_key"]
    result = ask_question_tool.invoke(input={'question':input_data.question})
    assert isinstance(result, str)

def test_run_with_none_key(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class 
    with a None session key.
    '''
    ask_question_tool.st_session_key = None
    result = ask_question_tool.invoke(input={'question':input_data.question})
    assert isinstance(result, str)
    ask_question_tool.sys_bio_model = ModelData()
    result = ask_question_tool.invoke(input={'question':input_data.question})
    # No model data or object in the streeamlit key
    assert result == "Please provide a valid model object or \
                    Streamlit session key that contains the model object."
    # delete the session key form the session state
    del st.session_state["test_key"]
    ask_question_tool.st_session_key = "test_key"
    result = ask_question_tool.invoke(input={'question':input_data.question})
    expected_result = f"Session key {ask_question_tool.st_session_key} "
    expected_result += "not found in Streamlit session state."
    assert result == expected_result

def test_run_with_a_simulated_model(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class 
    with a valid session key and model data.
    '''
    model = BasicoModel(model_id=64)
    model.simulate(duration=2, interval=2)
    ask_question_tool.sys_bio_model = ModelData(model_object=model)
    result = ask_question_tool.invoke(input={'question':input_data.question})
    assert isinstance(result, str)

def test_get_metadata(ask_question_tool):
    '''
    Test the get_metadata method of the AskQuestionTool class.
    '''
    metadata = ask_question_tool.get_metadata()
    assert metadata["name"] == "ask_question"
    assert metadata["description"] == "A tool to ask question about the simulation results."
