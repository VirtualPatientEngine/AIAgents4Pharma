'''
Test cases
'''

from langchain_core.messages import HumanMessage, ToolMessage
from ..agents.t2b_agent import get_app

def test_get_modelinfo_tool():
    '''
    Test the get_modelinfo tool.
    '''
    unique_id = 12345
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    app.update_state(config,{"sbml_file_path": ["BIOMD0000000537.xml"]})
    prompt = "Extract all relevant information from the uploaded model."
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)

def test_search_models_tool():
    '''
    Test the search_models tool.
    '''
    unique_id = 12345
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    prompt = "Search for models on Crohn's disease."
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)
    # Check if the assistant message contains the
    # biomodel id BIO0000000537
    assert "BIOMD0000000537" in assistant_msg

def test_ask_question_tool():
    '''
    Test the ask_question tool without the simulation results.
    '''
    unique_id = 12345
    app = get_app(unique_id, llm_model='gpt-4o-mini')
    config = {"configurable": {"thread_id": unique_id}}

    ##########################################
    # Test ask_question tool when simulation
    # results are not available
    ##########################################
    # Update state
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    prompt = "Call the ask_question tool to answer the "
    prompt += "question: What is the concentration of CRP "
    prompt += "in serum at 1000 hours?"

    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)

def test_simulate_model_tool():
    '''
    Test the simulate_model tool.
    '''
    unique_id = 123
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    # ##########################################
    # ## Test simulate_model tool
    # ##########################################
    prompt = "Simulate the model 537 for 2016 hours and intervals"
    prompt += " 2016 with an initial concentration of `DoseQ2W` "
    prompt += "set to 300 and `Dose` set to 0. Reset the concentration"
    prompt += " of `NAD` to 100 every 500 hours."
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
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    prompt = "What is the concentration of CRP in serum at 1000 hours? "
    # prompt += "Show only the concentration, rounded to 1 decimal place."
    # prompt += "For example, if the concentration is 0.123456, "
    # prompt += "your response should be `0.1`. Do not return any other information."
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # print (assistant_msg)
    # Check if the assistant message is a string
    assert "1.7" in assistant_msg

    ##########################################
    # Test custom_plotter tool when the
    # simulation results are available
    ##########################################
    prompt = "Plot only CRP related species."

    # Update state
    app.update_state(config, {"llm_model": "gpt-4o-mini"}
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
    expected_artifact = ['CRP[serum]', 'CRPExtracellular']
    expected_artifact += ['CRP Suppression (%)', 'CRP (% of baseline)']
    expected_artifact += ['CRP[liver]']
    predicted_artifact = []
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage):
            # Work on the message if it is a ToolMessage
            # These may contain additional visuals that
            # need to be displayed to the user.
            if msg.name == "custom_plotter":
                predicted_artifact = msg.artifact
                break
    # Check if the two artifacts are equal
    # assert expected_artifact in predicted_artifact
    assert set(expected_artifact).issubset(set(predicted_artifact))
    ##########################################
    # Test custom_plotter tool when the
    # simulation results are available but
    # the species is not available
    ##########################################
    prompt = "Plot the species `TP53`."

    # Update state
    app.update_state(config, {"llm_model": "gpt-4o-mini"}
                    )
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # print (response["messages"])
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
