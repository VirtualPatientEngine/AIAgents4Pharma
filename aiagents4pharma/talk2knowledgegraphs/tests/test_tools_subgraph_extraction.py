"""
Test cases for tools/subgraph_extraction.py
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
        "uploaded_files": [],
        "topk_nodes": 3,
        "topk_edges": 3,
        "dic_source_graph": [
            {
                "name": "PrimeKG",
                "kg_pyg_path": f"{DATA_PATH}/primekg_ibd_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/primekg_ibd_text_graph.pkl",
            }
        ],
    }

    return input_dict


def test_extract_subgraph_wo_docs(input_dict):
    """
    Test the subgraph extraction tool without any documents using OpenAI model.

    Args:
        input_dict: Input dictionary.
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
    Please directly invoke `subgraph_extraction` tool without calling any other tools 
    to respond to the following prompt:

    Extract all relevant information related to nodes of genes related to inflammatory bowel disease 
    (IBD) that existed in the knowledge graph.
    Please set the extraction name for this process as `subkg_12345`.
    """

    # Test the tool subgraph_extraction
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check tool message
    tool_msg = response["messages"][-2]
    assert tool_msg.name == "subgraph_extraction"

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
    # Check if the nodes are in the graph_text
    assert all(
        n[0] in dic_extracted_graph["graph_text"]
        for n in dic_extracted_graph["graph_dict"]["nodes"]
    )
    # Check if the edges are in the graph_text
    assert all(
        ",".join([e[0], '"' + str(tuple(e[2]["relation"])) + '"', e[1]])
        in dic_extracted_graph["graph_text"]
        for e in dic_extracted_graph["graph_dict"]["edges"]
    )


def test_extract_subgraph_w_docs(input_dict):
    """
    Test the subgraph extraction tool with a document as reference (i.e., endotype document)
    using OpenAI model.

    Args:
        input_dict: Input dictionary.
    """
    # Prepare LLM and embedding model
    input_dict["llm_model"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    input_dict["embedding_model"] = OpenAIEmbeddings(model="text-embedding-3-small")

    # Setup the app
    unique_id = 12345
    app = get_app(unique_id, llm_model=input_dict["llm_model"])
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    input_dict["uploaded_files"] = [
        {
            "file_name": "DGE_human_Colon_UC-vs-Colon_Control.pdf",
            "file_path": f"{DATA_PATH}/DGE_human_Colon_UC-vs-Colon_Control.pdf",
            "file_type": "endotype",
            "uploaded_by": "VPEUser",
            "uploaded_timestamp": "2024-11-05 00:00:00",
        }
    ]
    app.update_state(
        config,
        input_dict,
    )
    prompt = """
    Please ONLY invoke `subgraph_extraction` tool without calling any other tools 
    to respond to the following prompt:

    Extract all relevant information related to nodes of genes related to inflammatory bowel disease 
    (IBD) that existed in the knowledge graph.
    Please set the extraction name for this process as `subkg_12345`.
    """

    # Test the tool subgraph_extraction
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check tool message
    tool_msg = response["messages"][-2]
    assert tool_msg.name == "subgraph_extraction"

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
    # Check if the nodes are in the graph_text
    assert all(
        n[0] in dic_extracted_graph["graph_text"]
        for n in dic_extracted_graph["graph_dict"]["nodes"]
    )
    # Check if the edges are in the graph_text
    assert all(
        ",".join([e[0], '"' + str(tuple(e[2]["relation"])) + '"', e[1]])
        in dic_extracted_graph["graph_text"]
        for e in dic_extracted_graph["graph_dict"]["edges"]
    )
