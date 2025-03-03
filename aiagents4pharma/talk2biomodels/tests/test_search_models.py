'''
Test cases for Talk2Biomodels search models tool.
'''

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

def test_search_models_tool():
    '''
    Test the search_models tool.
    '''
    unique_id = 12345
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Search for models on Crohn's disease."
    app.update_state(
            config,
            {"llm_model": LLM_MODEL}
        )
    # Test the tool get_modelinfo
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    # Extract the assistant artifact which contains
    # all the search results
    found_model_537 = False
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "search_models":
            msg_artifact = msg.artifact
            for model in msg_artifact["dic_data"]:
                if model["id"] == "BIOMD0000000537":
                    found_model_537 = True
            break
    # Check if the model BIOMD0000000537 is found
    assert found_model_537
