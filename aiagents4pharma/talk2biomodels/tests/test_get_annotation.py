'''
Test cases for Talk2Biomodels.
'''

import random
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from ..agents.t2b_agent import get_app
from ..api.uniprot import search_uniprot_labels
from ..api.ols import fetch_from_ols
from ..api.kegg import fetch_kegg_names, fetch_from_api

@pytest.fixture(name="make_graph")
def make_graph_fixture():
    '''
    Create an instance of the talk2biomodels agent.
    '''
    unique_id = random.randint(1000, 9999)
    graph = get_app(unique_id)
    config = {"configurable": {"thread_id": unique_id}}
    return graph, config

def test_species_list(make_graph):
    '''
    Test the tool by passing species names.
    '''
    # Test with a valid species name
    app, config = make_graph
    prompt = "Extract annotations of species IL6 in model 537."
    app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
    current_state = app.get_state(config)
    print (current_state.values["dic_annotations_data"])
    dic_annotations_data = current_state.values["dic_annotations_data"]
    assert dic_annotations_data[0]['data']["Species Name"][0] == "IL6"

    # Test with an invalid species name
    app, config = make_graph
    prompt = "Extract annotations of species NADH in model 537."
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    artifact_was_none = True
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            if msg.artifact is None and 'NADH' in msg.content:
                artifact_was_none = False
                break
    assert not artifact_was_none

    # Test with an invalid species name and a valid species name
    app, config = make_graph
    prompt = "Extract annotations of species NADH, NAD, and IL7 in model 64."
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    print (current_state.values["dic_annotations_data"])
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    artifact_was_none = False
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            print (msg.artifact, msg.content)
            if msg.artifact is True and 'IL7' in msg.content:
                artifact_was_none = True
                break
    assert artifact_was_none

def test_all_species(make_graph):
    '''
    Test the tool by asking for annotations of all species is specific models.

    model 12 contains species with no URL.
    model 20 contains species without description.
    model 56 contains species with database outside of KEGG, UniProt, and OLS.
    '''
    for model_id in [12, 20, 56]:
        app, config = make_graph
        prompt = f"Extract annotations of all species model {model_id}."
        # Test the tool get_modelinfo
        response = app.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config
                        )
        assistant_msg = response["messages"][-1].content
        print (assistant_msg)
        current_state = app.get_state(config)
        dic_annotations_data = current_state.values["dic_annotations_data"]
        print (dic_annotations_data)
        assert isinstance(dic_annotations_data, list)

# In the following test cases, we will test the API functions,
# which were not tested in the cases above.

def test_search_uniprot_labels():
    '''
    Test the search_uniprot_labels function.
    '''
    # Test with a valid identifier and an invalid identifier
    identifiers = ["P61764", "P0000Q"]
    results = search_uniprot_labels(identifiers)
    assert results["P61764"] == "Syntaxin-binding protein 1"
    assert results["P0000Q"].startswith("Error: 400")

def test_fetch_from_ols():
    '''
    Test the fetch_from_ols function.
    '''
    # Enter a term that does not exist
    term = "GO:ABC123"
    label = fetch_from_ols(term)
    assert label.startswith("Error: 404")

def test_fetch_kegg_names():
    '''
    Test the fetch_kegg_names function.
    '''
    # Test with valid IDs
    ids = ["C00001", "C00002"]
    results = fetch_kegg_names(ids)
    assert results["C00001"] == "H2O"
    assert results["C00002"] == "ATP"

    # Test with an empty list
    results = fetch_kegg_names([])
    assert not results

def test_fetch_from_api():
    '''
    Test the fetch_from_api function.
    '''
    # Test with a valid query
    base_url = "https://rest.kegg.jp/get/"
    query = "C00001"
    entry_data = fetch_from_api(base_url, query)
    assert entry_data.startswith("ENTRY       C00001")

    # Test with an invalid query
    query = "C0000Q"
    entry_data = fetch_from_api(base_url, query)
    assert not entry_data
