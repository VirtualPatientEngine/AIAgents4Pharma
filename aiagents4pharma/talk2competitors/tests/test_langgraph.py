'''
Test cases
'''

from langchain_core.messages import HumanMessage
from ..agents.main_agent import get_app

def test_main_agent():
    '''
    Test the main agent.
    '''
    unique_id = "test_12345"
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    ####################################################
    prompt = "Without calling any tool, tell me the capital of France"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert 'Paris' in assistant_msg
    ####################################################
    prompt = "Search articles on machine learning"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert 'Fashion-MNIST' in assistant_msg
    ####################################################
    prompt = "Recommend articles using the first paper of the previous search"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    assistant_msg = response["messages"][-1].content
    print (assistant_msg)
    # Check if the assistant message is a string
    assert 'CNN Models' in assistant_msg
    ####################################################
    prompt = "Recommend articles using both papers of your last response"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    assistant_msg = response["messages"][-1].content
    print (assistant_msg)
    # Check if the assistant message is a string
    assert 'Efficient Handwritten Digit Classification' in assistant_msg
    ###################################################
    prompt = "Show me the papers in the state"
    response = app.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "is_last_step": False,
            "current_agent": None,
        },
        config=config,
    )

    assistant_msg = response["messages"][-1].content
    print (assistant_msg)
    # Check if the assistant message is a string
    assert 'Classification of Fashion-MNIST Dataset' in assistant_msg
