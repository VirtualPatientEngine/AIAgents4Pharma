'''
Test Talk2AIAgents4Pharma supervisor agent.
'''

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ..agents.main_agent import get_app

# Define the data path for the test files of Talk2KnowledgeGraphs agent
DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"
LLM_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

@pytest.fixture(name="input_dict")
def input_dict_fixture():
    """
    Input dictionary fixture for Talk2AIAgents4Pharma agent,
    which is partly inherited from the Talk2KnowledgeGraphs agent.
    """
    input_dict = {
        "topk_nodes": 3,
        "topk_edges": 3,
        "uploaded_files": [],
        "dic_source_graph": [
            {
                "name": "PrimeKG",
                "kg_pyg_path": f"{DATA_PATH}/primekg_ibd_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/primekg_ibd_text_graph.pkl",
            }
        ],
        "dic_extracted_graph": []
    }

    return input_dict

def test_main_agent_invokes_t2kg(input_dict):
    """
    In the following test, we will ask the main agent (supervisor)
    to list drugs that target the gene Interleukin-6. We will check
    if the Talk2KnowledgeGraphs agent is invoked. We will do so by
    checking the state of the Talk2AIAgents4Pharma agent, which is
    partly inherited from the Talk2KnowledgeGraphs agent

    Args:
        input_dict: Input dictionary
    """
    # Prepare LLM and embedding model
    input_dict["llm_model"] = LLM_MODEL
    input_dict["embedding_model"] = OpenAIEmbeddings(model="text-embedding-3-small")

    # Setup the app
    unique_id = 12345
    app = get_app(unique_id, llm_model=input_dict["llm_model"])
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    app.update_state(
        config,
        input_dict,
    )
    prompt = "List drugs that target the gene Interleukin-6"

    # Invoke the agent
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check extracted subgraph dictionary
    current_state = app.get_state(config)
    dic_extracted_graph = current_state.values["dic_extracted_graph"][0]
    assert isinstance(dic_extracted_graph, dict)
    assert dic_extracted_graph["graph_source"] == "PrimeKG"
    assert dic_extracted_graph["topk_nodes"] == 3
    assert dic_extracted_graph["topk_edges"] == 3
    assert isinstance(dic_extracted_graph["graph_dict"], dict)
    assert len(dic_extracted_graph["graph_dict"]["nodes"]) > 0
    assert len(dic_extracted_graph["graph_dict"]["edges"]) > 0
    assert isinstance(dic_extracted_graph["graph_text"], str)
    # Check summarized subgraph
    assert isinstance(dic_extracted_graph["graph_summary"], str)

def test_main_agent_invokes_t2b():
    '''
    In the following test, we will ask the main agent (supervisor)
    to simulate a model. And we will check if the Talk2BioModels
    agent is invoked. We will do so by checking the state of the
    Talk2AIAgents4Pharma agent, which is partly inherited from the
    Talk2BioModels agent.
    '''
    unique_id = 123
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "Simulate model 64"
    # Invoke the agent
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    # Get the state of the Talk2AIAgents4Pharma agent
    current_state = app.get_state(config)
    # Check if the dic_simulated_data is in the state
    dic_simulated_data = current_state.values["dic_simulated_data"]
    # Check if the dic_simulated_data is a list
    assert isinstance(dic_simulated_data, list)
    # Check if the length of the dic_simulated_data is 1
    assert len(dic_simulated_data) == 1
    # Check if the source of the model is 64
    assert dic_simulated_data[0]['source'] == 64
    # Check if the data of the model contains
    # '1,3-bisphosphoglycerate'
    assert '1,3-bisphosphoglycerate' in dic_simulated_data[0]['data']
