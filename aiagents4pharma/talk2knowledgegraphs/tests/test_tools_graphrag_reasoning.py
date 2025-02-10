"""
Test cases for tools/graphrag_reasoning.py
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
        "graph_summary": """
            The summarized subgraph focuses on **immune responses** and their regulation,
            particularly in relation to **inflammation**. Here are the key elements highlighted:

            ### Important Nodes:
            - **NOD2:** A gene involved in immune responses, linked to Crohn's disease through
            genetic mutations associated with inflammation.
            - **IL10:** An anti-inflammatory cytokine that regulates immune responses,
            particularly limiting the host's immune response to pathogens.
            - **TLR4:** A receptor for lipopolysaccharides from Gram-negative bacteria,
            triggering innate immune responses and playing a role in inflammation
            and infection defense.
            - **Th17 Activation Pathway:** A subset of CD4+ T helper cells that produce IL-17,
            crucial in autoimmune diseases like multiple sclerosis and rheumatoid arthritis.
            - **Autophagy:** A cellular process responsible for recycling and removing damaged
            components, maintaining cellular homeostasis and preventing disease.
            - **Inflammasome Pathway:** A multi-protein complex that activates inflammatory
            processes through the cleavage of pro-inflammatory cytokines like IL-1β.

            ### Important Edges:
            - **g5 to p5:** TLR4 is a gene member of the IL-27 signaling pathway, crucial for
            regulating immune responses.
            - **g5 to p6:** TLR4 is a gene member of the Inflammasome pathway,
            which activates inflammatory processes.
            - **g4 to p10:** IL10 is a gene member of the Th17 activation pathway,
            significant in autoimmune diseases.
            - **g4 to p5:** IL10 is a gene member of the IL-27 signaling pathway,
            important for regulating immune responses.
            - **g5 to p1:** Autophagy is a cellular process that plays a crucial role in maintaining
            cellular homeostasis and preventing disease.

            ### Summary:
            This subgraph emphasizes the interactions among various genes and pathways that regulate
            immune responses, particularly in the context of inflammation.
            It highlights the roles of NOD2, IL10, TLR4, the Th17 activation pathway, autophagy,
            and the inflammasome pathway in immune regulation.
            """,
    }

    return input_dict


def test_graphrag_reasoning_openai(input_dict):
    """
    Test the GraphRAG reasoning tool using OpenAI model.

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
    Please invoke `graphrag_reasoning` tool without calling any other tools 
    to respond to the following prompt:

    Without extracting a new subgraph, perform Graph RAG reasoning 
    to get insights related to nodes of genes mentioned in the knowledge graph related to DrugA. 

    DrugA is a human monoclonal antibody that binds to both 
    the soluble and transmembrane bioactive forms of human TNFa (UniProt Acc: P01375).
    """

    # Test the tool  graphrag_reasoning
    response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Check assistant message
    assistant_msg = response["messages"][-1].content
    assert isinstance(assistant_msg, str)

    # Check tool message
    tool_msg = response["messages"][-2]
    assert tool_msg.name == "graphrag_reasoning"

    # Check reasoning results
    assert "DrugA" in assistant_msg
    assert "TNF" in assistant_msg
