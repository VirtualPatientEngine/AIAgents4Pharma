'''
Test cases for Talk2Biomodels.
'''

import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

def test_integration():
    '''
    Test the integration of the tools.
    '''
    unique_id = 1234567
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    # ##########################################
    # ## Test simulate_model tool
    # ##########################################
    prompt = '''Simulate the model BIOMD0000000537 for 100 hours and time intervals
    100 with an initial concentration of `DoseQ2W` set to 300 and `Dose`
    set to 0. Reset the concentration of `Ab{serum}` to 100 every 25 hours.'''
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    print (assistant_msg)
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)
    ##########################################
    # Test ask_question tool when simulation
    # results are available
    ##########################################
    # Update state
    app.update_state(config, {"llm_model": LLM_MODEL})
    prompt = """What is the concentration of CRP in serum after 100 hours?
    Round off the value to 2 decimal places."""
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # print (assistant_msg)
    # Check if the assistant message is a string
    assert '211' in assistant_msg

    ##########################################
    # Test the custom_plotter tool when the
    # simulation results are available but
    # the species is not available
    ##########################################
    prompt = """Call the custom_plotter tool to make a plot
        showing only species 'Infected cases'. Let me
        know if these species were not found. Do not
        invoke any other tool."""
    # Update state
    app.update_state(config, {"llm_model": LLM_MODEL}
                    )
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    current_state = app.get_state(config)
    # Get the messages from the current state
    # and reverse the order
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    predicted_artifact = []
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage):
            # Work on the message if it is a ToolMessage
            # These may contain additional visuals that
            # need to be displayed to the user.
            if msg.name == "custom_plotter":
                predicted_artifact = msg.artifact
                break
    # Check if the the predicted artifact is `None`
    assert predicted_artifact is None

    ##########################################
    # Test custom_plotter tool when the
    # simulation results are available
    ##########################################
    prompt = "Plot only CRP related species."

    # Update state
    app.update_state(config, {"llm_model": LLM_MODEL}
                    )
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    current_state = app.get_state(config)
    # Get the messages from the current state
    # and reverse the order
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages
    # until a ToolMessage is found.
    expected_header = ['Time', 'CRP{serum}', 'CRPExtracellular']
    expected_header += ['CRP Suppression (%)', 'CRP (% of baseline)']
    expected_header += ['CRP{liver}']
    predicted_artifact = []
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage):
            # Work on the message if it is a ToolMessage
            # These may contain additional visuals that
            # need to be displayed to the user.
            if msg.name == "custom_plotter":
                predicted_artifact = msg.artifact['dic_data']
                break
    # Convert the artifact into a pandas dataframe
    # for easy comparison
    df = pd.DataFrame(predicted_artifact)
    # Extract the headers from the dataframe
    predicted_header = df.columns.tolist()
    # Check if the header is in the expected_header
    # assert expected_header in predicted_artifact
    assert set(expected_header).issubset(set(predicted_header))
