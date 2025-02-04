'''
Test cases for Talk2Biomodels.
'''

import pandas as pd
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
    # results are not available i.e. the
    # simulation has not been run. In this
    # case, the tool should return an error
    ##########################################
    # Update state
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    # Define the prompt
    prompt = "Call the ask_question tool to answer the "
    prompt += "question: What is the concentration of CRP "
    prompt += "in serum at 1000 hours? The simulation name "
    prompt += "is `simulation_name`."
    # Invoke the tool
    app.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )
    # Get the messages from the current state
    # and reverse the order
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    for msg in reversed_messages:
        # Assert that the message is a ToolMessage
        # and its status is "error"
        if isinstance(msg, ToolMessage):
            assert msg.status == "error"

def test_simulate_model_tool():
    '''
    Test the simulate_model tool when simulating
    multiple models.
    '''
    unique_id = 123
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    # Upload a model to the state
    app.update_state(config,
        {"sbml_file_path": ["aiagents4pharma/talk2biomodels/tests/BIOMD0000000449_url.xml"]})
    prompt = "Simulate model 64 and the uploaded model"
    # Invoke the agent
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    dic_simulated_data = current_state.values["dic_simulated_data"]
    # Check if the dic_simulated_data is a list
    assert isinstance(dic_simulated_data, list)
    # Check if the length of the dic_simulated_data is 2
    assert len(dic_simulated_data) == 2
    # Check if the source of the first model is 64
    assert dic_simulated_data[0]['source'] == 64
    # Check if the source of the second model is upload
    assert dic_simulated_data[1]['source'] == "upload"
    # Check if the data of the first model contains
    assert '1,3-bisphosphoglycerate' in dic_simulated_data[0]['data']
    # Check if the data of the second model contains
    assert 'mTORC2' in dic_simulated_data[1]['data']

def test_param_scan_tool():
    '''
    In this test, we will test the parameter_scan tool.
    We will prompt it to scan the parameter `kIL6RBind`
    from 1 to 100 in steps of 10, record the changes
    in the concentration of the species `Ab{serum}` in
    model 537.

    We will pass the inaccuarate parameter (`KIL6Rbind`)
    and species names (just `Ab`) to the tool to test
    if it can deal with it.

    We expect the agent to first invoke the parameter_scan
    tool and raise an error. It will then invoke another
    tool get_modelinfo to get the correct parameter
    and species names. Finally, the agent will reinvoke
    the parameter_scan tool with the correct parameter
    and species names.

    '''
    unique_id = 123
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    prompt = """How will the value of Ab in model 537 change
            if the param kIL6Rbind is varied from 1 to 100 in steps of 10?
            Set the initial `DoseQ2W` concentration to 300. Also, reset
            the IL6{serum} concentration to 100 every 500 hours and assume
            that the model is simulated for 2016 hours with an interval of 2016."""
    # Invoke the agent
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    df = pd.DataFrame(columns=['name', 'status', 'content'])
    names = []
    statuses = []
    contents = []
    for msg in reversed_messages:
        # Assert that the message is a ToolMessage
        # and its status is "error"
        if not isinstance(msg, ToolMessage):
            continue
        names.append(msg.name)
        statuses.append(msg.status)
        contents.append(msg.content)
    df = pd.DataFrame({'name': names, 'status': statuses, 'content': contents})
    # print (df)
    assert any((df["status"] == "error") &
               (df["name"] == "parameter_scan") &
               (df["content"].str.startswith("Error: ValueError('Invalid parameter name:")))
    assert any((df["status"] == "success") &
               (df["name"] == "parameter_scan") &
               (df["content"].str.startswith("Parameter scan results of")))
    assert any((df["status"] == "success") &
               (df["name"] == "get_modelinfo"))

def test_steady_state_tool():
    '''
    Test the steady_state tool.
    '''
    unique_id = 123
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(config, {"llm_model": "gpt-4o-mini"})
    #########################################################
    # In this case, we will test if the tool returns an error
    # when the model does not achieve a steady state. The tool
    # status should be "error".
    prompt = """Run a steady state analysis of model 537."""
    # Invoke the agent
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    tool_msg_status = None
    for msg in reversed_messages:
        # Assert that the status of the
        # ToolMessage is "error"
        if isinstance(msg, ToolMessage):
            # print (msg)
            tool_msg_status = msg.status
            break
    assert tool_msg_status == "error"
    #########################################################
    # In this case, we will test if the tool is indeed invoked
    # successfully
    prompt = """Run a steady state analysis of model 64.
    Set the initial concentration of `Pyruvate` to 0.2. The
    concentration of `NAD` resets to 100 every 2 time units."""
    # Invoke the agent
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    # Loop through the reversed messages until a
    # ToolMessage is found.
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    steady_state_invoked = False
    for msg in reversed_messages:
        # Assert that the message is a ToolMessage
        # and its status is "error"
        if isinstance(msg, ToolMessage):
            print (msg)
            if msg.name == "steady_state" and msg.status != "error":
                steady_state_invoked = True
                break
    assert steady_state_invoked
    #########################################################
    # In this case, we will test if the `ask_question` tool is
    # invoked upon asking a question about the already generated
    # steady state results
    prompt = """What is the Phosphoenolpyruvate concentration
        at the steady state? Show onlyconcentration, rounded to
        2 decimal places. For example, if the concentration is
        0.123456, your response should be `0.12`. Do not return
        any other information."""
    # Invoke the agent
    response = app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    assistant_msg = response["messages"][-1].content
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    ask_questool_invoked = False
    for msg in reversed_messages:
        # Assert that the message is a ToolMessage
        # and its status is "error"
        if isinstance(msg, ToolMessage):
            if msg.name == "ask_question":
                ask_questool_invoked = True
                break
    assert ask_questool_invoked
    assert "0.06" in assistant_msg

def test_integration():
    '''
    Test the integration of the tools.
    '''
    unique_id = 1234567
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
    prompt = """What is the concentration of CRP
            in serum after 1000 time points?"""
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
                predicted_artifact = msg.artifact
                break
    # Convert the artifact into a pandas dataframe
    # for easy comparison
    df = pd.DataFrame(predicted_artifact)
    # Extract the headers from the dataframe
    predicted_header = df.columns.tolist()
    # Check if the header is in the expected_header
    # assert expected_header in predicted_artifact
    assert set(expected_header).issubset(set(predicted_header))
    ##########################################
    # Test custom_plotter tool when the
    # simulation results are available but
    # the species is not available
    ##########################################
    prompt = """Make a custom plot showing the
        concentration of the species `TP53` over
        time. Do not show any other species."""
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
