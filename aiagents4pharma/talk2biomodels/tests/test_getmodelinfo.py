'''
Test cases for Talk2Biomodels get_modelinfo tool.
'''

from langchain_core.messages import HumanMessage
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
