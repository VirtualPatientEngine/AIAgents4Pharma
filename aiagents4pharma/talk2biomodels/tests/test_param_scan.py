'''
Test cases for Talk2Biomodels parameter scan tool.
'''

import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

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
    unique_id = 1234
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = """How will the value of Ab in serum in model 537 change
            if the param kIL6Rbind is varied from 1 to 100 in steps of 10?
            Set the initial `DoseQ2W` concentration to 300. Assume
            that the model is simulated for 2016 hours with an interval of 50."""
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
               (df["content"].str.startswith(
                   "Error: ValueError('Invalid species or parameter name:")))
    assert any((df["status"] == "success") &
               (df["name"] == "parameter_scan") &
               (df["content"].str.startswith("Parameter scan results of")))
    assert any((df["status"] == "success") &
               (df["name"] == "get_modelinfo"))
