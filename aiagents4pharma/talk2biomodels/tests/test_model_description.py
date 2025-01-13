'''
Test cases for model_description.py
'''

import pytest
import streamlit as st
from ..tools.model_description import ModelDescriptionInput, ModelDescriptionTool, ModelData
from ..models.basico_model import BasicoModel

@pytest.fixture(name="model_description_tool")
def model_description_tool_fixture():
    '''
    Fixture for creating an instance of ModelDescriptionTool.
    '''
    return ModelDescriptionTool(st_session_key="test_key")

@pytest.fixture(name="input_data")
def input_data_fixture():
    '''
    Fixture for creating an instance of AskQuestionInput.
    '''
    return ModelDescriptionInput(question="Describe the model",
                            sys_bio_model=ModelData(model_id=64))

@pytest.fixture(name="basico_model", scope="module")
def basico_model_fixture():
    '''
    Fixture for creating an instance of BasicoModel.
    '''
    return BasicoModel(model_id=64)

def test_run_with_missing_session_key(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class 
    with a missing session key.
    '''
    if 'test_key' in st.session_state:
        del st.session_state['test_key']
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        })
    expected_result = f"Session key {model_description_tool.st_session_key} "
    expected_result += "not found in Streamlit session state."
    assert result == expected_result

def test_check_model_object(basico_model):
    """
    Test the check_model_object method of the ModelData class.
    """
    # Test with valid BasicoModel object
    model_data = ModelData(model_object=basico_model)
    validated_data = model_data.check_model_object(model_data.__dict__)
    assert validated_data['model_object'] is not None

    # Test with invalid model_object
    model_data = ModelData(model_object="invalid_object")
    validated_data = model_data.check_model_object(model_data.__dict__)
    assert validated_data['model_object'] is None

    # Test with None model_object
    model_data = ModelData(model_object=None)
    validated_data = model_data.check_model_object(model_data.__dict__)
    assert validated_data['model_object'] is None

def test_run_with_none_key_no_model_data(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a valid session key.
    '''
    st.session_state["test_key"] = None
    input_data.sys_bio_model = ModelData()
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert result == "Please provide a BioModels ID or an SBML file path for the model."

def test_call_run_with_different_model_data(input_data, basico_model, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a model id.
    '''
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert isinstance(result, str)
    # Test the _run method of the ModelDescriptionTool class with an SBML file.
    input_data = ModelDescriptionInput(question="Describe the model",
        sys_bio_model=ModelData(
            sbml_file_path="aiagents4pharma/talk2biomodels/tests/BIOMD0000000064_url.xml"),
        st_session_key="test_key"
        )
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert isinstance(result, str)
    # Test the _run method of the ModelDescriptionTool class with a model object.
    input_data = ModelDescriptionInput(question="Describe the model",
                            sys_bio_model=ModelData(model_object=basico_model),
                            st_session_key="test_key"
                            )
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert isinstance(result, str)

def test_run_with_none_key(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a None
    '''
    model_description_tool.st_session_key = None
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert isinstance(result, str)
    # sleep for 5 seconds
    # time.sleep(5)
    input_data.sys_bio_model = ModelData()
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert result == "Please provide a valid model object or " \
                    "Streamlit session key that contains the model object."
    # sleep for 5 seconds
    # time.sleep(5)
    model_description_tool.st_session_key = "test_key"
    # delete the session key form the session state
    del st.session_state[model_description_tool.st_session_key]
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert result == f"Session key {model_description_tool.st_session_key} " \
        "not found in Streamlit session state."

def test_run_manager(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a run_manager.
    '''
    model_description_tool.metadata = {"prompt": '''Given: {description},
                                                    answer the question:
                                                    {question}.'''}
    result = model_description_tool.invoke(input={
                                        'question':input_data.question,
                                        'sys_bio_model':input_data.sys_bio_model,
                                        })
    assert isinstance(result, str)

def test_get_metadata(model_description_tool):
    '''
    Test the get_metadata method of the ModelDescriptionTool class.
    '''
    metadata = model_description_tool.get_metadata()
    assert metadata["name"] == "model_description"
    assert metadata["description"] == "A tool to ask about the description of the model."
