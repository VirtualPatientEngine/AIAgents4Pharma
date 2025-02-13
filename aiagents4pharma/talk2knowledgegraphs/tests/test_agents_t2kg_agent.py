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
                "file_name": "adalimumab.pdf",
                "file_path": f"{DATA_PATH}/adalimumab.pdf",
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
        "topk_nodes": 3,
        "topk_edges": 3,
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
    Adalimumab is a fully human monoclonal antibody (IgG1) 
    that specifically binds to tumor necrosis factor-alpha (TNF-α), a pro-inflammatory cytokine.

    I would like to get evidence from the knowledge graph about the mechanism of actions related to
    Adalimumab in treating inflammatory bowel disease 
    (IBD). Please follow these steps:
    - Extract a subgraph from the PrimeKG that contains information about Adalimumab.
    - Summarize the extracted subgraph.
    - Reason about the mechanism of action of Adalimumab in treating IBD.

    Please set the extraction name for the extraction process as `subkg_12345`.
    """

    # Test the tool get_modelinfo
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check extracted subgraph dictionary
    current_state = app.get_state(config)
    dic_extracted_graph = current_state.values["dic_extracted_graph"][0]
    assert isinstance(dic_extracted_graph, dict)
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "PrimeKG"
    assert dic_extracted_graph["topk_nodes"] == 3
    assert dic_extracted_graph["topk_edges"] == 3
    assert isinstance(dic_extracted_graph["graph_dict"], dict)
    assert len(dic_extracted_graph["graph_dict"]["nodes"]) > 0
    assert len(dic_extracted_graph["graph_dict"]["edges"]) > 0
    assert isinstance(dic_extracted_graph["graph_text"], str)
    # Check summarized subgraph
    assert isinstance(dic_extracted_graph["graph_summary"], str)
    # Check reasoning results
    assert "Adalimumab" in assistant_msg
    assert "TNF" in assistant_msg
