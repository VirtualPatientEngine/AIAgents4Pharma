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
        "selected_genes": [],
        "selected_drugs": [],
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


def test_extract_multimodal_subgraph_text(input_dict):
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


def test_extract_multimodal_subgraph_genes(input_dict):
    """
    Test the multimodal subgraph extraction tool for text as modality, plus genes.

    Args:
        input_dict: Input dictionary.
    """
    # Add a selected genes to the input dictionary
    input_dict["selected_genes"] = ["IL6_(1567)", "IL21_(34967)", "TNF_(2329)"]

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

    Extract all relevant information related to nodes of genes related to inflammatory bowel disease (IBD) that existed in the knowledge graph.
    In particular, extract a set of nodes and edges interconnecting the IL6, IL21, and TNF genes
    for potentially involved in the
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
    # Check if the selected genes are in the graph_dict
    assert all(
        gene in [n[0] for n in dic_extracted_graph["graph_dict"]['nodes']]
        for gene in input_dict["selected_genes"]
    )
    # Check wheter the selected genes are in the graph_text
    assert all(
        gene in dic_extracted_graph["graph_text"]
        for gene in input_dict["selected_genes"]
    )

def test_extract_multimodal_subgraph_drugs(input_dict):
    """
    Test the multimodal subgraph extraction tool for text as modality, plus drugs.

    Args:
        input_dict: Input dictionary.
    """
    # Add a selected drugs to the input dictionary
    input_dict["selected_drugs"] = ["Mesalazine_(15876)"]

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

    Extract all relevant information related to nodes of genes related
    to inflammatory bowel disease (IBD) that existed in the knowledge graph.

    In particular, extract a set of nodes and edges interconnecting toward a drug called
    Mesalazine and Betamethasone for potentially treating the disease.
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
    # Check if the selected drugs are in the graph_dict
    assert all(
        drug in [n[0] for n in dic_extracted_graph["graph_dict"]['nodes']]
        for drug in input_dict["selected_drugs"]
    )
    # Check wheter the selected drugs are in the graph_text
    assert all(
        drug in dic_extracted_graph["graph_text"]
        for drug in input_dict["selected_drugs"]
    )
