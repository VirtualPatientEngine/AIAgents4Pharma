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
        "llm_model": ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        "embedding_model": OpenAIEmbeddings(model="text-embedding-3-small"),
        "selections": {
            "gene/protein": [],
            "molecular_function": [],
            "cellular_component": [],
            "biological_process": [],
            "drug": [],
            "disease": []
        },
        "uploaded_files": [],
        "topk_nodes": 3,
        "topk_edges": 3,
        "dic_source_graph": [
            {
                "name": "PrimeKG",
                "kg_pyg_path": f"{DATA_PATH}/biobridge_multimodal_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/biobridge_multimodal_text_graph.pkl",
            }
        ],
    }

    return input_dict


def test_extract_multimodal_subgraph_wo_doc(input_dict):
    """
    Test the multimodal subgraph extraction tool for only text as modality.

    Args:
        input_dict: Input dictionary.
    """
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
    As a knowledge graph agent, I would like you to call a tool called `subgraph_extraction`.
    After calling the tool, restrain yourself to call any other tool.

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
        n[0] in dic_extracted_graph["graph_text"].replace('"', '')
        for n in dic_extracted_graph["graph_dict"]["nodes"]
    )
    # Check if the edges are in the graph_text
    assert all(
        ",".join([e[0], str(tuple(e[2]["relation"])), e[1]])
        in dic_extracted_graph["graph_text"].replace('"', '')
        for e in dic_extracted_graph["graph_dict"]["edges"]
    )


def test_extract_multimodal_subgraph_w_doc(input_dict):
    """
    Test the multimodal subgraph extraction tool for text as modality, plus genes.

    Args:
        input_dict: Input dictionary.
    """
    # Add a selected genes to the input dictionary
    input_dict["uploaded_files"] = [
        {
            "file_name": "multimodal-analysis.xlsx",
            "file_path": f"{DATA_PATH}/multimodal-analysis.xlsx",
            "file_type": "multimodal",
            "uploaded_by": "VPEUser",
            "uploaded_timestamp": "2025-05-12 00:00:00",
        }
    ]

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
    As a knowledge graph agent, I would like you to call a tool called `subgraph_extraction`.
    After calling the tool, restrain yourself to call any other tool.

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
        n[0] in dic_extracted_graph["graph_text"].replace('"', '')
        for n in dic_extracted_graph["graph_dict"]["nodes"]
    )
    # Check if the edges are in the graph_text
    assert all(
        ",".join([e[0], str(tuple(e[2]["relation"])), e[1]])
        in dic_extracted_graph["graph_text"].replace('"', '')
        for e in dic_extracted_graph["graph_dict"]["edges"]
    )
    # Check if the selected concepts are in the graph_dict
    assert all(
        i in [n[0] for n in dic_extracted_graph["graph_dict"]['nodes']]
        for k,v in response["selections"].items()
        for i in v
    )
    # Check wheter the selected concepts are in the graph_text
    assert all(
        i in dic_extracted_graph["graph_text"]
        for k,v in response["selections"].items()
        for i in v
    )
