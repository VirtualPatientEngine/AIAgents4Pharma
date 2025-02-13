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
        "dic_extracted_graph": [
            {
                "name": "subkg_12345",
                "tool_call_id": "tool_12345",
                "graph_source": "PrimeKG",
                "topk_nodes": 3,
                "topk_edges": 3,
                "graph_dict": {
                    'nodes': [('IFNG_(3495)', {}), 
                              ('IKBKG_(3672)', {}),
                              ('ATG16L1_(6661)', {}),
                              ('inflammatory bowel disease_(28158)', {}),
                              ('Crohn ileitis and jejunitis_(35814)', {}),
                              ("Crohn's colitis_(83770)", {})],
                    'edges': [('IFNG_(3495)', 'inflammatory bowel disease_(28158)', 
                               {'relation': ['gene/protein', 'associated with', 'disease'],
                                'label': ['gene/protein', 'associated with', 'disease']}), 
                              ('IFNG_(3495)', "Crohn's colitis_(83770)",
                               {'relation': ['gene/protein', 'associated with', 'disease'],
                                'label': ['gene/protein', 'associated with', 'disease']}),
                              ('IFNG_(3495)', 'Crohn ileitis and jejunitis_(35814)',
                               {'relation': ['gene/protein', 'associated with', 'disease'],
                                'label': ['gene/protein', 'associated with', 'disease']}), 
                              ('ATG16L1_(6661)', 'IKBKG_(3672)',
                               {'relation': ['gene/protein', 'ppi', 'gene/protein'],
                                'label': ['gene/protein', 'ppi', 'gene/protein']}), 
                              ("Crohn's colitis_(83770)", 'ATG16L1_(6661)',
                               {'relation': ['disease', 'associated with', 'gene/protein'],
                                'label': ['disease', 'associated with', 'gene/protein']})]},
                "graph_text": """
            node_id,node_attr
            IFNG_(3495),"IFNG belongs to gene/protein category. 
            This gene encodes a soluble cytokine that is a member of the type II interferon class. 
            The encoded protein is secreted by cells of both the innate and adaptive immune systems. 
            The active protein is a homodimer that binds to the interferon gamma receptor 
            which triggers a cellular response to viral and microbial infections. 
            Mutations in this gene are associated with an increased susceptibility to viral, 
            bacterial and parasitic infections and to several autoimmune diseases. 
            [provided by RefSeq, Dec 2015]."
            IKBKG_(3672),"IKBKG belongs to gene/protein category. This gene encodes the regulatory 
            subunit of the inhibitor of kappaB kinase (IKK) complex, which activates NF-kappaB 
            resulting in activation of genes involved in inflammation, immunity, cell survival, 
            and other pathways. Mutations in this gene result in incontinentia pigmenti, 
            hypohidrotic ectodermal dysplasia, and several other types of immunodeficiencies. 
            A pseudogene highly similar to this locus is located in an adjacent region of the 
            X chromosome. [provided by RefSeq, Mar 2016]."
            ATG16L1_(6661),"ATG16L1 belongs to gene/protein category. The protein encoded 
            by this gene is part of a large protein complex that is necessary for autophagy, 
            the major process by which intracellular components are targeted to lysosomes 
            for degradation. Defects in this gene are a cause of susceptibility to inflammatory 
            bowel disease type 10 (IBD10). Several transcript variants encoding different 
            isoforms have been found for this gene.[provided by RefSeq, Jun 2010]."
            inflammatory bowel disease_(28158),inflammatory bowel disease belongs to disease 
            category. Any inflammatory bowel disease in which the cause of the disease 
            is a mutation in the NOD2 gene.  
            Crohn ileitis and jejunitis_(35814),Crohn ileitis and jejunitis belongs to 
            disease category. An Crohn disease involving a pathogenic inflammatory 
            response in the ileum.  
            Crohn's colitis_(83770),Crohn's colitis belongs to disease category. 
            Crohn's disease affecting the colon.  

            head_id,edge_type,tail_id
            Crohn's colitis_(83770),"('disease', 'associated with', 'gene/protein')",
            ATG16L1_(6661)
            ATG16L1_(6661),"('gene/protein', 'ppi', 'gene/protein')",IKBKG_(3672)
            IFNG_(3495),"('gene/protein', 'associated with', 'disease')",
            inflammatory bowel disease_(28158)
            IFNG_(3495),"('gene/protein', 'associated with', 'disease')",Crohn's colitis_(83770)
            IFNG_(3495),"('gene/protein', 'associated with', 'disease')",
            Crohn ileitis and jejunitis_(35814)
            """,
                "graph_summary": """
            The subgraph extracted from `subkg_12345` includes several important genes and 
            their associations with inflammatory bowel diseases, particularly Crohn's disease.

            Key Nodes:
            1. **IFNG (Interferon gamma)**: This gene encodes a cytokine that plays a crucial 
            role in immune response. It is associated with several diseases, including 
            inflammatory bowel disease and specifically Crohn's colitis and Crohn ileitis and 
            jejunitis. Mutations in IFNG can lead to increased susceptibility to infections 
            and autoimmune diseases.

            2. **IKBKG (Inhibitor of kappaB kinase gamma)**: This gene is involved in the 
            regulation of NF-kappaB, which is critical for inflammation and immune responses. 
            Mutations can lead to immunodeficiencies and other disorders.

            3. **ATG16L1**: This gene is essential for autophagy, a process that helps in 
            degrading intracellular components. Defects in ATG16L1 are linked to inflammatory 
            bowel disease type 10 (IBD10) and are associated with Crohn's colitis.

            4. **Inflammatory Bowel Disease**: A category of diseases characterized by 
            chronic inflammation of the gastrointestinal tract, with specific mention of 
            mutations in the NOD2 gene as a cause.

            5. **Crohn's Colitis**: A specific type of Crohn's disease affecting the colon, 
            indicating a pathogenic inflammatory response.

            6. **Crohn Ileitis and Jejunitis**: Another form of Crohn's disease that involves 
            inflammation in the ileum.

            Key Edges:
            - **IFNG is associated with inflammatory bowel disease, Crohn's colitis, and 
            Crohn ileitis and jejunitis**: This highlights the role of IFNG in these diseases.
            - **ATG16L1 is associated with Crohn's colitis**: This indicates a direct link 
            between the gene and the disease.
            - **ATG16L1 interacts with IKBKG**: This protein-protein interaction suggests a 
            functional relationship between these two genes in the context of immune response 
            and inflammation.

            In summary, the subgraph illustrates the connections between key genes 
            (IFNG, IKBKG, ATG16L1) and their associations with inflammatory bowel diseases, 
            particularly Crohn's disease, emphasizing the genetic underpinnings of these conditions.
            """,
            }
        ],
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
    Without extracting a new subgraph, based on subgraph extracted from `subkg_12345`
    perform Graph RAG reasoning to get insights related to nodes of genes 
    mentioned in the knowledge graph related to Adalimumab. 

    Here is an additional context:
    Adalimumab is a fully human monoclonal antibody (IgG1) 
    that specifically binds to tumor necrosis factor-alpha (TNF-α), a pro-inflammatory cytokine.
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
    assert "Adalimumab" in assistant_msg
    assert "TNF" in assistant_msg
