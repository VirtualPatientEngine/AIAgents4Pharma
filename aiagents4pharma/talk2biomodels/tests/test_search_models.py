'''
Test cases for Talk2Biomodels search models tool.
'''

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ..agents.t2b_agent import get_app

def test_search_models_tool():
    '''
    Test the search_models tool.
    '''
    unique_id = 12345
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    app.update_state(config,
            {"llm_model": ChatNVIDIA(model="meta/llama-3.3-70b-instruct")})
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
