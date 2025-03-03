'''
Test cases for Talk2Biomodels get_annotation tool.
'''

import random
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..agents.t2b_agent import get_app
from ..tools.get_annotation import prepare_content_msg

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

@pytest.fixture(name="make_graph")
def make_graph_fixture():
    '''
    Create an instance of the talk2biomodels agent.
    '''
    unique_id = random.randint(1000, 9999)
    graph = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    graph.update_state(
            config,
            {"llm_model": LLM_MODEL}
        )
    return graph, config

def test_no_model_provided(make_graph):
    '''
    Test the tool by not specifying any model.
    We are testing a condition where the user
    asks for annotations of all species without
    specifying a model.
    '''
    app, config = make_graph
    prompt = "Extract annotations of all species. Call the tool get_annotation."
    app.invoke({"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    current_state = app.get_state(config)
    # Assert that the state key model_id is empty.
    assert current_state.values["model_id"] == []

def test_valid_species_provided(make_graph):
    '''
    Test the tool by providing a specific species name.
    We are testing a condition where the user asks for annotations
    of a specific species in a specific model.
    '''
    # Test with a valid species name
    app, config = make_graph
    prompt = "Extract annotations of species IL6 in model 537."
    app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
    current_state = app.get_state(config)
    # print (current_state.values["dic_annotations_data"])
    dic_annotations_data = current_state.values["dic_annotations_data"]

    # The assert statement checks if IL6 is present in the returned annotations.
    assert dic_annotations_data[0]['data']["Species Name"][0] == "IL6"

def test_invalid_species_provided(make_graph):
    '''
    Test the tool by providing an invalid species name.
    We are testing a condition where the user asks for annotations
    of an invalid species in a specific model.
    '''
    # Test with an invalid species name
    app, config = make_graph
    prompt = "Extract annotations of only species NADH in model 537."
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.

    test_condition = False
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            #If a ToolMessage exists and artifact is None (meaning no valid annotation was found)
            #and the rejected species (NADH) is mentioned, the test passes.
            if msg.artifact is None and msg.status == "error":
                #If artifact is None, it means no annotation was found
                # (likely due to an invalid species).
                test_condition = True
                break
    assert test_condition

def test_invalid_and_valid_species_provided(make_graph):
    '''
    Test the tool by providing an invalid species name and a valid species name.
    We are testing a condition where the user asks for annotations
    of an invalid species and a valid species in a specific model.
    '''
    # Test with an invalid species name and a valid species name
    app, config = make_graph
    prompt = "Extract annotations of species NADH, NAD, and IL7 in model 64."
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    dic_annotations_data = current_state.values["dic_annotations_data"]
    # List of species that are expected to be found in the annotations
    extracted_species = []
    for idx in dic_annotations_data[0]['data']["Species Name"]:
        extracted_species.append(dic_annotations_data[0]['data']["Species Name"][idx])
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    tool_status_success = False
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            if msg.artifact is True and msg.status == "success":
                tool_status_success = True
                break
    assert tool_status_success
    assert set(extracted_species) == set(["NADH", "NAD"])

def test_all_species_annotations(make_graph):
    '''
    Test the tool by asking for annotations of all species is specific models.
    Here, we test the tool with three models since they have different use cases:
        - model 12 contains a species with no URL provided.
        - model 20 contains a species without description.
        - model 56 contains a species with database outside of KEGG, UniProt, and OLS.

    We are testing a condition where the user asks for annotations
    of all species in a specific model.
    '''
    # Loop through the models and test the tool
    # for each model's unique use case.
    for model_id in [12, 20, 56]:
        app, config = make_graph
        prompt = f"Extract annotations of all species model {model_id}."
        # Test the tool get_modelinfo
        app.invoke({"messages": [HumanMessage(content=prompt)]},
                            config=config
                        )
        current_state = app.get_state(config)

        reversed_messages = current_state.values["messages"][::-1]
        # Coveres all of the use cases for the expecetd sting on all the species
        test_condition = False
        for msg in reversed_messages:
            # Skip messages that are not ToolMessages and those that are not
            # from the get_annotation tool.
            if not isinstance(msg, ToolMessage) or msg.name != "get_annotation":
                continue
            if model_id == 12:
                # Extact the first and second description of the LacI protein
                # We already know that the first or second description is missing ('-')
                dic_annotations_data = current_state.values["dic_annotations_data"][0]
                first_descp_laci_protein = dic_annotations_data['data']['Description'][0]
                second_descp_laci_protein = dic_annotations_data['data']['Description'][1]

                # Expect a successful extraction (artifact is True) and that the content
                # matches what is returned by prepare_content_msg for species.
                # And that the first or second description of the LacI protein is missing.
                if (msg.artifact is True and msg.content == prepare_content_msg([])
                    and msg.status=="success" and (first_descp_laci_protein == '-' or
                                                    second_descp_laci_protein == '-')):
                    test_condition = True
                    break

            if model_id == 20:
                # Expect an error message containing a note
                # that species extraction failed.
                if ("Unable to extract species from the model"
                    in msg.content and msg.status == "error"):
                    test_condition = True
                    break

            if model_id == 56:
                # Expect a successful extraction (artifact is True) and that the content
                # matches for for missing description ['ORI'].
                if (msg.artifact is True and
                msg.content == prepare_content_msg(['ORI'])
                and msg.status == "success"):
                    test_condition = True
                    break
        assert test_condition # Expected output is validated
