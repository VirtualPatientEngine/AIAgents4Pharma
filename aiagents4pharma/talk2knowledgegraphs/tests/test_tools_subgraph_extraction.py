"""
Test cases for tools/subgraph_extraction.py
"""

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
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
        "input_tkg": f"{DATA_PATH}/kg_pyg_graph.pkl",
        "input_text_tkg": f"{DATA_PATH}/kg_text_graph.pkl",
        "topk_nodes": 3,
        "topk_edges": 3,
    }

    return input_dict


def test_extract_subgraph_wo_docs(input_dict):
    """
    Test the subgraph extraction tool without any documents using Ollama model.

    Args:
        input_dict: Input dictionary.
    """
    # Prepare LLM and embedding model
    input_dict["llm_model"] = ChatOllama(model="llama3.2:1b", temperature=0.0)
    input_dict["embedding_model"] = OllamaEmbeddings(model="nomic-embed-text")

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
    Please ONLY invoke `subgraph_extraction` tool without calling any other tools 
    to respond to the following prompt:

    Extract all relevant information related to nodes of genes existed in the knowledge graph.
    """

    # Test the tool subgraph_extraction
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check tool message
    tool_msg = response["messages"][-2]
    assert tool_msg.name == "subgraph_extraction"

    # Check extracted subgraph
    assert isinstance(response["graph_dict"], dict)
    assert len(response["graph_dict"]["nodes"]) > 0
    assert len(response["graph_dict"]["edges"]) > 0
    assert isinstance(response["graph_text"], str)
    # Check if the nodes are in the graph_text
    assert all(n[0] in response["graph_text"] for n in response["graph_dict"]["nodes"])
    # Check if the edges are in the graph_text
    assert all(
        ",".join([e[0], '"' + str(tuple(e[2]["relation"])) + '"', e[1]])
        in response["graph_text"]
        for e in response["graph_dict"]["edges"]
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

    Extract all relevant information related to nodes of genes existed in the knowledge graph.
    """

    # Test the tool subgraph_extraction
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check tool message
    tool_msg = response["messages"][-2]
    assert tool_msg.name == "subgraph_extraction"

    # Check extracted subgraph
    assert isinstance(response["graph_dict"], dict)
    assert len(response["graph_dict"]["nodes"]) > 0
    assert len(response["graph_dict"]["edges"]) > 0
    assert isinstance(response["graph_text"], str)
    # Check if the nodes are in the graph_text
    assert all(n[0] in response["graph_text"] for n in response["graph_dict"]["nodes"])
    # Check if the edges are in the graph_text
    assert all(
        ",".join([e[0], '"' + str(tuple(e[2]["relation"])) + '"', e[1]])
        in response["graph_text"]
        for e in response["graph_dict"]["edges"]
    )
