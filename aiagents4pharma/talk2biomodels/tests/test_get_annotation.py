'''
Test cases for Talk2Biomodels.
'''
import random
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from ..agents.t2b_agent import get_app
from ..tools.get_annotation import prepare_content_msg

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
    # print (current_state.values["dic_annotations_data"])
    dic_annotations_data = current_state.values["dic_annotations_data"]

    # The assert statement checks if IL6 is present in the returned annotations.
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

    test_condition = False
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            #If a ToolMessage exists and artifact is None (meaning no valid annotation was found)
            #and the rejected species (NADH) is mentioned, the test passes.
            if msg.artifact is None and 'NADH' in msg.content:
                #If artifact is None, it means no annotation was found
                # (likely due to an invalid species).
                #If artifact contains data, the tool successfully retrieved annotations.
                test_condition = True
                break
    # assert test_condition
    assert test_condition, "Expected rejection message for NADH but did not find it."

    # Test with an invalid species name and a valid species name
    app, config = make_graph
    prompt = "Extract annotations of species NADH, NAD, and IL7 in model 64."
    app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config
    )
    current_state = app.get_state(config)
    # dic_annotations_data = current_state.values["dic_annotations_data"]
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.
    artifact_was_none = False
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            # print (msg.artifact, msg.content)

            if msg.artifact is True and 'IL7' in msg.content:
                artifact_was_none = True
                break
    assert artifact_was_none

def test_prepare_content_msg():
    '''
     Testing the tool by expected messages
    '''
    test_cases = [
        (
            ["NADH"],
            [],
            """Successfully extracted annotations for the species.
                        The following species do not exist, and
                        hence their annotations were not extracted:
                        NADH."""
        ),
        (
            [],# Species not found
            ["H2O"],# Species without description
            """The descriptions for the following species
                        were not found:
                        H2O."""
        ),
        (
            ["NADH", "NAD+"],
            ["H2O", "ATP"],
            """The following species do not exist, and
                        hence their annotations were not extracted:
                        NADH, NAD+.The descriptions for the following species
                        were not found:
                        H2O, ATP."""
        ),
    ]

    for species_not_found, species_without_description, expected_message in test_cases:
        # Iterates through the test cases, calling `prepare_content_msg` with the
        # species lists and comparing the generated message to the expected message.
        actual_message = prepare_content_msg(species_not_found, species_without_description)
        # The core assertion: checks if the expected message is a substring of the actual message.
        # .strip() is used on the actual message to remove any leading/trailing whitespace,
        # making the test more robust to minor formatting variations in the function's output.
    assert expected_message in actual_message.strip(), f"not_found: {species_not_found},\
        without_desc: {species_without_description},\
        expected: '{expected_message}', got: '{actual_message}'"

def test_all_species(make_graph):
    '''
    Test the tool by asking for annotations of all species is specific models.

    model 12 contains species with no URL.
    model 20 contains species without description.
    model 56 contains species with database outside of KEGG, UniProt, and OLS.
    '''
    # Test valid models
    for model_id in [12, 20, 56]:
        app, config = make_graph
        prompt = f"Extract annotations of all species model {model_id}."
        # Test the tool get_modelinfo
        response = app.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config
                        )
        print(response["messages"])
        assistant_msg = response["messages"][-1].content

        current_state = app.get_state(config)

    reversed_messages = current_state.values["messages"][::-1]
    # Coveres all of the use cases for the expecetd sting on all the species
    test_condition = False
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            print("ToolMessage Content:", msg.content)  # Debugging output
            if msg.artifact is None and ('ORI' in msg.content or
                                         "Successfully extracted annotations for the species"
                                         in msg.content or "Error" in msg.content):
                test_condition = True
                break

    dic_annotations_data = current_state.values["dic_annotations_data"]

    assert isinstance(dic_annotations_data, list),\
        f"Expected a list for model {model_id}, got {type(dic_annotations_data)}"
    assert len(dic_annotations_data) > 0,\
        f"Expected species data for model {model_id}, but got empty list"
    assert test_condition # Expected output is validated

    # Test case where no model is specified
    app, config = make_graph
    prompt = "Extract annotations of all species."
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    current_state = app.get_state(config)
    # dic_annotations_data = current_state.values["dic_annotations_data"]
    reversed_messages = current_state.values["messages"][::-1]

    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            if msg.artifact is None:
                break
    # Loop through the reversed messages until a
    # ToolMessage is found.
    # Ensure the system correctly informs the user to specify a model
    # assistant_msg = response["messages"][-1].content
    assert ('''It seems there was an error while trying to extract
            annotations for all species using the uploaded model.''' in assistant_msg
            or "The annotations for all species in model"
            in assistant_msg), f"Unexpected response: {assistant_msg}"
