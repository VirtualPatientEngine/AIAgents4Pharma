"""
Test cases for Talk2Biomodels.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def test_save_model_tool():
    """
    Test the save_model tool.
    """
    unique_id = 123
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    # Simulate a model
    prompt = "Simulate model 64"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Save a model
    prompt = "Save the model"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Simulate and save a model
    prompt = "Simulate model 64 and then save the model at /home/gsingh/"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Simulate and ave a model
    prompt = "Simulate model 64 and then save the model at /xyz/"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Simulate and ave a model
    prompt = "Simulate model 64 and then save the model at None"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # dic_simulated_data = current_state.values["dic_simulated_data"]
    # # Check if the dic_simulated_data is a list
    # assert isinstance(dic_simulated_data, list)
    # # Check if the length of the dic_simulated_data is 2
    # assert len(dic_simulated_data) == 2
    # # Check if the source of the first model is 64
    # assert dic_simulated_data[0]["source"] == 64
    # # Check if the source of the second model is upload
    # assert dic_simulated_data[1]["source"] == "upload"
    # # Check if the data of the first model contains
    # assert "1,3-bisphosphoglycerate" in dic_simulated_data[0]["data"]
    # # Check if the data of the second model contains
    # assert "mTORC2" in dic_simulated_data[1]["data"]
    # # Check if the model_as_string is not None
    # assert current_state.values["model_as_string"][-1] is not None
