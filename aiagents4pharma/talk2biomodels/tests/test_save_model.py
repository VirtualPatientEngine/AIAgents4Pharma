"""
Test cases for Talk2Biomodels.
"""

import tempfile

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
    # Save a model without simulating
    prompt = "Save the model"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
    # Create a temporary directory to save the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save a model to the temporary directory
        prompt = f"Simulate model 64 and save it model at {temp_dir}"
        # Invoke the agent
        app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        current_state = app.get_state(config)
        assert current_state.values["model_as_string"][-1] is not None
    # Simulate and save a model in non-existing path
    prompt = "Simulate model 64 and then save the model at /xyz/"
    # Invoke the agent
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    assert current_state.values["model_as_string"][-1] is not None
