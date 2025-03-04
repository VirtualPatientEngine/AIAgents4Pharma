'''
Test cases for Talk2Biomodels get_modelinfo tool.
'''

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini',temperature=0)

def test_get_modelinfo_tool():
    '''
    Test the get_modelinfo tool.
    '''
    unique_id = 12345
    app = get_app(unique_id, LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    app.update_state(config,
      {"sbml_file_path": ["aiagents4pharma/talk2biomodels/tests/BIOMD0000000449_url.xml"]})
    prompt = "Extract all relevant information from the uploaded model."
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)

def test_model_with_no_species():
    '''
    Test the get_modelinfo tool with a model that does not
    return any species.

    This should raise a tool error.
    '''
    unique_id = 12345
    app = get_app(unique_id, LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Extract all species from model 20"
    # Test the tool get_modelinfo
    app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    test_condition = False
    for msg in reversed_messages:
        # Check if the message is a ToolMessage from the get_modelinfo tool
        if isinstance(msg, ToolMessage) and msg.name == "get_modelinfo":
            # Check if the message is an error message
            if (msg.status == "error" and
                "ValueError('Unable to extract species from the model.')" in msg.content):
                test_condition = True
                break
    assert test_condition

def test_model_with_no_parameters():
    '''
    Test the get_modelinfo tool with a model that does not
    return any parameters.

    This should raise a tool error.
    '''
    unique_id = 12345
    app = get_app(unique_id, LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Extract all parameters from model 10"
    # Test the tool get_modelinfo
    app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    test_condition = False
    for msg in reversed_messages:
        # Check if the message is a ToolMessage from the get_modelinfo tool
        if isinstance(msg, ToolMessage) and msg.name == "get_modelinfo":
            # Check if the message is an error message
            if (msg.status == "error" and
                "ValueError('Unable to extract parameters from the model.')" in msg.content):
                test_condition = True
                break
    assert test_condition
