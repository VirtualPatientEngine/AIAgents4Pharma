"""
Test cases for tools/subgraph_extraction.py
"""

from langchain_core.messages import HumanMessage
from ..agents.t2kg_agent import get_app


def test_extract_subgraph_wo_docs():
    """
    Test the subgraph extraction tool without any documents.
    """
    # Setup the app
    unique_id = 12345
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    data_path = "aiagents4pharma/talk2knowledgegraphs/tests/files"
    app.update_state(
        config,
        {
            "llm_model": "gpt-4o-mini",
            "uploaded_files": [],
            "input_tkg": f"{data_path}/kg_pyg_graph.pkl",
            "input_text_tkg": f"{data_path}/kg_text_graph.pkl",
            "topk_nodes": 3,
            "topk_edges": 3,
        },
    )
    prompt = """
    Extract all relevant information from the uploaded model.
    Please only use `subgraph_extraction` tool.
    """

    # Test the tool get_modelinfo
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


def test_extract_subgraph_w_docs():
    """
    Test the subgraph extraction tool with a document as reference (i.e., endotype document).
    """
    # Setup the app
    unique_id = 12345
    app = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state
    data_path = "aiagents4pharma/talk2knowledgegraphs/tests/files"
    app.update_state(
        config,
        {
            "llm_model": "gpt-4o-mini",
            "uploaded_files": [
                {
                    "file_name": "DGE_human_Colon_UC-vs-Colon_Control.pdf",
                    "file_path": f"{data_path}/DGE_human_Colon_UC-vs-Colon_Control.pdf",
                    "file_type": "endotype",
                    "uploaded_by": "VPEUser",
                    "uploaded_timestamp": "2024-11-05 00:00:00",
                }
            ],
            "input_tkg": f"{data_path}/kg_pyg_graph.pkl",
            "input_text_tkg": f"{data_path}/kg_text_graph.pkl",
            "topk_nodes": 3,
            "topk_edges": 3,
        },
    )
    prompt = """
    Extract all relevant information from the uploaded model.
    Please only use `subgraph_extraction` tool.
    """

    # Test the tool get_modelinfo
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
