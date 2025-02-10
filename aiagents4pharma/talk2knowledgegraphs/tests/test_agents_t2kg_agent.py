"""
Test cases for agents/t2kg_agent.py
"""

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ..agents.t2kg_agent import get_app

# Define the data path
DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"


@pytest.fixture(name="input_dict")
def input_dict_fixture():
    """
    Input dictionary fixture.
    """
    input_dict = {
        "llm_model": None,  # TBA for each test case
        "embedding_model": None,  # TBA for each test case
        "uploaded_files": [
            {
                "file_name": "DrugA.pdf",
                "file_path": f"{DATA_PATH}/DrugA.pdf",
                "file_type": "drug_data",
                "uploaded_by": "VPEUser",
                "uploaded_timestamp": "2024-11-05 00:00:00",
            },
            {
                "file_name": "DGE_human_Colon_UC-vs-Colon_Control.pdf",
                "file_path": f"{DATA_PATH}/DGE_human_Colon_UC-vs-Colon_Control.pdf",
                "file_type": "endotype",
                "uploaded_by": "VPEUser",
                "uploaded_timestamp": "2024-11-05 00:00:00",
            },
        ],
        "input_tkg": f"{DATA_PATH}/kg_pyg_graph.pkl",
        "input_text_tkg": f"{DATA_PATH}/kg_text_graph.pkl",
        "topk_nodes": 3,
        "topk_edges": 3,
        "graph_dict": None,
        "graph_text": None,
        "graph_summary": None,
    }

    return input_dict


def test_t2kg_agent_openai(input_dict):
    """
    Test the T2KG agent using OpenAI model.

    Args:
        input_dict: Input dictionary
    """
    # Prepare LLM and embedding model
    input_dict["llm_model"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
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
    prompt = """
    DrugA is a human monoclonal antibody that binds to both
    the soluble and transmembrane bioactive forms of human TNFa (UniProt Acc: P01375).

    I would like to get evidence from the knowledge graph about the mechanism of actions related to
    DrugA. Make sure to highlights key nodes and edges in the extracted subgraph.
    Make sure to discover insights related to TNF and its interactions with other entities.
    Perform reasoning on the extracted subgraph to generate a concise
    summary of the mechanism of action of DrugA.
    """

    # Test the tool get_modelinfo
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check reasoning results
    assert "DrugA" in assistant_msg
    assert "TNF" in assistant_msg
