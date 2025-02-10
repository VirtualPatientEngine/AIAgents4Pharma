"""
Test cases for tools/subgraph_summarization.py
"""

import pytest
from langchain_core.messages import HumanMessage
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
        "graph_dict": {
            "nodes": [
                ("g1", {}),
                ("g4", {}),
                ("g5", {}),
                ("p5", {}),
                ("p10", {}),
                ("p1", {}),
                ("p6", {}),
            ],
            "edges": [
                (
                    "g1",
                    "p1",
                    {
                        "relation": ["gene", "member_of", "pathway"],
                        "label": ["gene", "member_of", "pathway"],
                    },
                ),
                (
                    "g4",
                    "p10",
                    {
                        "relation": ["gene", "member_of", "pathway"],
                        "label": ["gene", "member_of", "pathway"],
                    },
                ),
                (
                    "g4",
                    "p5",
                    {
                        "relation": ["gene", "member_of", "pathway"],
                        "label": ["gene", "member_of", "pathway"],
                    },
                ),
                (
                    "g5",
                    "p5",
                    {
                        "relation": ["gene", "member_of", "pathway"],
                        "label": ["gene", "member_of", "pathway"],
                    },
                ),
                (
                    "g5",
                    "p6",
                    {
                        "relation": ["gene", "member_of", "pathway"],
                        "label": ["gene", "member_of", "pathway"],
                    },
                ),
                (
                    "g5",
                    "p1",
                    {
                        "relation": ["gene", "member_of", "pathway"],
                        "label": ["gene", "member_of", "pathway"],
                    },
                ),
            ],
        },
        "graph_text": """
            node_id,node_attr
            g1,"NOD2 is a gene that contributes to immune responses and has been implicated in
            Crohn's disease, particularly through genetic mutations linked to inflammation."
            g4,"IL10 is an anti-inflammatory cytokine that regulates immune responses,
            particularly in limiting host immune response to pathogens to prevent
            damage to the body."
            g5,"TLR4 is a receptor for lipopolysaccharides from Gram-negative bacteria,
            which triggers an innate immune response and plays a role in inflammation
            and infection defense."
            p5,"IL-27 signaling pathway plays a crucial role in regulating immune responses,
            particularly in the context of chronic inflammation and autoimmunity."
            p10,"Th17 Activation Pathway refers to a subset of CD4+ T helper cells that produce
            IL-17 cytokine, playing a crucial role in autoimmune diseases such as
            multiple sclerosis and rheumatoid arthritis."
            p1,"Autophagy is a cellular process responsible for recycling and removing damaged or
            dysfunctional cellular components, playing a crucial role in maintaining cellular
            homeostasis and preventing disease."
            p6,Inflammasome pathway refers to the multi-protein complexes that activate
            inflammatory processes through the cleavage of pro-inflammatory cytokines like IL-1β.

            source,edge_attr,target
            g5,"('gene', 'member_of', 'pathway')",p5
            g5,"('gene', 'member_of', 'pathway')",p6
            g4,"('gene', 'member_of', 'pathway')",p10
            g1,"('gene', 'member_of', 'pathway')",p1
            g4,"('gene', 'member_of', 'pathway')",p5
            g5,"('gene', 'member_of', 'pathway')",p1
            """,
    }

    return input_dict


def test_summarize_subgraph(input_dict):
    """
    Test the subgraph summarization tool without any documents using Ollama model.

    Args:
        input_dict: Input dictionary fixture.
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
    Please ONLY invoke `subgraph_summarization` tool without calling any other tools 
    to respond to the following prompt:

    You are given a subgraph in the forms of textualized subgraph representing
    nodes and edges (triples).
    Summarize the given subgraph and higlight the importance nodes and edges.
    """

    # Test the tool subgraph_summarization
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check tool message
    tool_msg = response["messages"][-2]
    assert tool_msg.name == "subgraph_summarization"

    # Check summarized subgraph
    assert isinstance(response["graph_summary"], str)
