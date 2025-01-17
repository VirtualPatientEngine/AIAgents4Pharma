'''
Test cases for the search_studies
'''

# from ..tools.search_studies import search_studies
from aiagents4pharma.talk2cells.agents.scp_agent import get_app
from langchain_core.messages import HumanMessage

def test_agent_scp():
    '''
    Test the agent_scp.
    '''
    unique_id = 12345
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Search for studies on Crohns Disease."
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    assistant_msg = response["messages"][-1].content
    # Check if the assistant message is a string
    assert isinstance(assistant_msg, str)
