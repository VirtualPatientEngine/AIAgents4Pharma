"""
Test cases for tools/subgraph_extraction.py
"""

import pytest
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ..tools.multimodal_subgraph_extraction import MultimodalSubgraphExtractionTool

# Define the data path
DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"


@pytest.fixture(name="agent_state")
def agent_state_fixture():
    """
    Agent state fixture.
    """
    agent_state = {
        # "llm_model": ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        # "embedding_model": OpenAIEmbeddings(model="text-embedding-3-small"),
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
                "name": "BioBridge",
                "kg_pyg_path": f"{DATA_PATH}/biobridge_multimodal_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/biobridge_multimodal_text_graph.pkl",
            }
        ],
    }

    return agent_state


def test_extract_multimodal_subgraph_wo_doc(agent_state):
    """
    Test the multimodal subgraph extraction tool for only text as modality.

    Args:
        agent_state: Agent state in the form of a dictionary.
    """
    prompt = """
    Extract all relevant information related to nodes of genes related to inflammatory bowel disease
    (IBD) that existed in the knowledge graph.
    Please set the extraction name for this process as `subkg_12345`.
    """

    # Instantiate the MultimodalSubgraphExtractionTool
    subgraph_extraction_tool = MultimodalSubgraphExtractionTool()

    # Invoking the subgraph_extraction_tool
    response = subgraph_extraction_tool.invoke(
        input={"prompt": prompt,
               "tool_call_id": "subgraph_extraction_tool",
               "state": agent_state,
               "arg_data": {"extraction_name": "subkg_12345"}})

    # Check tool message
    assert response.update["messages"][-1].tool_call_id  == "subgraph_extraction_tool"

    # Check extracted subgraph dictionary
    dic_extracted_graph = response.update["dic_extracted_graph"][0]
    assert isinstance(dic_extracted_graph, dict)
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "BioBridge"
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


def test_extract_multimodal_subgraph_w_doc(agent_state):
    """
    Test the multimodal subgraph extraction tool for text as modality, plus genes.

    Args:
        agent_state: Agent state in the form of a dictionary.
    """
    # Update state
    agent_state["uploaded_files"] = [
        {
            "file_name": "multimodal-analysis.xlsx",
            "file_path": f"{DATA_PATH}/multimodal-analysis.xlsx",
            "file_type": "multimodal",
            "uploaded_by": "VPEUser",
            "uploaded_timestamp": "2025-05-12 00:00:00",
        }
    ]

    prompt = """
    Extract all relevant information related to nodes of genes related to inflammatory bowel disease
    (IBD) that existed in the knowledge graph.
    Please set the extraction name for this process as `subkg_12345`.
    """

    # Instantiate the SubgraphExtractionTool
    subgraph_extraction_tool = MultimodalSubgraphExtractionTool()

    # Invoking the subgraph_extraction_tool
    response = subgraph_extraction_tool.invoke(
        input={"prompt": prompt,
               "tool_call_id": "subgraph_extraction_tool",
               "state": agent_state,
               "arg_data": {"extraction_name": "subkg_12345"}})

    # Check tool message
    assert response.update["messages"][-1].tool_call_id  == "subgraph_extraction_tool"

    # Check extracted subgraph dictionary
    dic_extracted_graph = response.update["dic_extracted_graph"][0]
    assert isinstance(dic_extracted_graph, dict)
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "BioBridge"
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
