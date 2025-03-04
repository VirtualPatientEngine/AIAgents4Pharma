'''
Test cases for Talk2Biomodels steady state tool.
'''

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

def test_steady_state_tool():
    '''
    Test the steady_state tool.
    '''
    unique_id = 123
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(
            config,
            {"llm_model": LLM_MODEL}
        )
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
    prompt = """Bring model 64 to a steady state. Set the
    initial concentration of `Pyruvate` to 0.2. The
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
        at the steady state? Show only the concentration, rounded
        to 2 decimal places. For example, if the concentration is
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
