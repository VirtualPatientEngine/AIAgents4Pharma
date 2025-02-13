"""
Test cases for tools/subgraph_summarization.py
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
                "graph_summary": None,
            }
        ],
    }

    return input_dict


def test_summarize_subgraph(input_dict):
    """
    Test the subgraph summarization tool without any documents using Ollama model.

    Args:
        input_dict: Input dictionary fixture.
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
    Please directly invoke `subgraph_summarization` tool without calling any other tools 
    to respond to the following prompt:

    You are given a subgraph in the forms of textualized subgraph representing
    nodes and edges (triples) obtained from extraction_name `subkg_12345`.
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
    current_state = app.get_state(config)
    dic_extracted_graph = current_state.values["dic_extracted_graph"][0]
    assert isinstance(dic_extracted_graph["graph_summary"], str)
