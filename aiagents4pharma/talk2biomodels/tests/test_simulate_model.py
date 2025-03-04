'''
Test cases for Talk2Biomodels.
'''

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

def test_simulate_model_tool():
    '''
    Test the simulate_model tool when simulating
    multiple models.
    '''
    unique_id = 123
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
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
