'''
Test cases for Talk2Biomodels.
'''

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

def test_ask_question_tool():
    '''
    Test the ask_question tool without the simulation results.
    '''
    unique_id = 12345
    app = get_app(unique_id, llm_model=ChatOpenAI(model='gpt-4o-mini', temperature=0))
    config = {"configurable": {"thread_id": unique_id}}

    ##########################################
    # Test ask_question tool when simulation
    # results are not available i.e. the
    # simulation has not been run. In this
    # case, the tool should return an error
    ##########################################
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
