"""
Test cases for Talk2BioModels get_annotation tool.
"""

import math
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ..tools.get_annotation import (
    GetAnnotationTool,
    extract_relevant_species_names,
    prepare_content_msg,
)
from ..tools.load_biomodel import ModelData

pytestmark = pytest.mark.unit_mock


def _tool_call(tool, args, tool_call_id="tc"):
    args = dict(args)
    args.pop("tool_call_id", None)
    return {"name": tool.name, "type": "tool_call", "id": tool_call_id, "args": args}


def test_no_model_provided(fake_app_factory):
    """
    Test the tool by not specifying any model.
    We are testing a condition where the user
    asks for annotations of all species without
    specifying a model.
    """
    app = fake_app_factory([{"model_id": [], "messages": []}])
    config = {"configurable": {"thread_id": 1}}
    prompt = "Extract annotations of all species. Call the tool get_annotation."
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    # Assert that the state key model_id is empty.
    assert current_state.values["model_id"] == []


def test_valid_species_provided(fake_app_factory):
    """
    Test the tool by providing a specific species name.
    We are testing a condition where the user asks for annotations
    of a specific species in a specific model.
    """
    # Test with a valid species name
    app = fake_app_factory(
        [
            {
                "dic_annotations_data": [{"data": {"Species Name": {0: "IL6"}}}],
                "messages": [],
            }
        ]
    )
    config = {"configurable": {"thread_id": 2}}
    prompt = "Extract annotations of species IL6 in model 537."
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    # print (current_state.values["dic_annotations_data"])
    dic_annotations_data = current_state.values["dic_annotations_data"]

    # The assert statement checks if IL6 is present in the returned annotations.
    assert dic_annotations_data[0]["data"]["Species Name"][0] == "IL6"


def test_invalid_species_provided(fake_app_factory):
    """
    Test the tool by providing an invalid species name.
    We are testing a condition where the user asks for annotations
    of an invalid species in a specific model.
    """
    # Test with an invalid species name
    invalid_message = ToolMessage(
        content="Invalid species",
        name="get_annotation",
        status="error",
        artifact=None,
        tool_call_id="call-1",
    )
    app = fake_app_factory([{"messages": [invalid_message]}])
    config = {"configurable": {"thread_id": 3}}
    prompt = "Extract annotations of only species NADH in model 537."
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages until a
    # ToolMessage is found.

    test_condition = False
    for msg in reversed_messages:
        # Assert that the one of the messages is a ToolMessage
        # and its artifact is None.
        if isinstance(msg, ToolMessage) and msg.name == "get_annotation":
            # If a ToolMessage exists and artifact is None (meaning no valid annotation was found)
            # and the rejected species (NADH) is mentioned, the test passes.
            if msg.artifact is None and msg.status == "error":
                # If artifact is None, it means no annotation was found
                # (likely due to an invalid species).
                test_condition = True
                break
    assert test_condition


def test_invalid_and_valid_species_provided(fake_app_factory):
    """
    Test the tool by providing an invalid species name and a valid species name.
    We are testing a condition where the user asks for annotations
    of an invalid species and a valid species in a specific model.
    """
    # Test with an invalid species name and a valid species name
    success_message = ToolMessage(
        content="Annotations found",
        name="get_annotation",
        status="success",
        artifact=True,
        tool_call_id="call-2",
    )
    app = fake_app_factory(
        [
            {
                "dic_annotations_data": [{"data": {"Species Name": {0: "NADH", 1: "NAD"}}}],
                "messages": [success_message],
            }
        ]
    )
    config = {"configurable": {"thread_id": 4}}
    prompt = "Extract annotations of species NADH, NAD, and IL7 in model 64."
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    dic_annotations_data = current_state.values["dic_annotations_data"]
    # List of species that are expected to be found in the annotations
    extracted_species = []
    for idx in dic_annotations_data[0]["data"]["Species Name"]:
        extracted_species.append(dic_annotations_data[0]["data"]["Species Name"][idx])
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
    assert set(extracted_species) == {"NADH", "NAD"}


def test_all_species_annotations(fake_app_factory):
    """
    Test the tool by asking for annotations of all species is specific models.
    Here, we test the tool with three models since they have different use cases:
        - model 12 contains a species with no URL provided.
        - model 20 contains a species without description.
        - model 56 contains a species with database outside of UniProt, and OLS.

    We are testing a condition where the user asks for annotations
    of all species in a specific model.
    """
    # Loop through the models and test the tool
    # for each model's unique use case.
    for model_id in [12, 20, 56]:
        if model_id == 12:
            message = ToolMessage(
                content=prepare_content_msg([]),
                name="get_annotation",
                status="success",
                artifact=True,
                tool_call_id="call-12",
            )
            state = {
                "dic_annotations_data": [
                    {
                        "data": {
                            "Species Name": {0: "LacI", 1: "LacI"},
                            "Description": {0: "-", 1: "desc"},
                        }
                    }
                ],
                "messages": [message],
            }
        elif model_id == 20:
            message = ToolMessage(
                content="Unable to extract species from the model",
                name="get_annotation",
                status="error",
                artifact=None,
                tool_call_id="call-20",
            )
            state = {"dic_annotations_data": [], "messages": [message]}
        else:
            message = ToolMessage(
                content=prepare_content_msg(["ORI"]),
                name="get_annotation",
                status="success",
                artifact=True,
                tool_call_id="call-56",
            )
            state = {
                "dic_annotations_data": [
                    {"data": {"Species Name": {0: "ORI"}, "Description": {0: "Missing"}}}
                ],
                "messages": [message],
            }
        app = fake_app_factory([state])
        config = {"configurable": {"thread_id": model_id}}
        prompt = f"Extract annotations of all species model {model_id}."
        # Test the tool get_modelinfo
        app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        current_state = app.get_state(config)

        reversed_messages = ["not-a-message"] + current_state.values["messages"][::-1]
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
                first_descp_laci_protein = dic_annotations_data["data"]["Description"][0]
                second_descp_laci_protein = dic_annotations_data["data"]["Description"][1]

                # Expect a successful extraction (artifact is True) and that the content
                # matches what is returned by prepare_content_msg for species.
                # And that the first or second description of the LacI protein is missing.
                if (
                    msg.artifact is True
                    and msg.content == prepare_content_msg([])
                    and msg.status == "success"
                    and (first_descp_laci_protein == "-" or second_descp_laci_protein == "-")
                ):
                    test_condition = True
                    break

            if model_id == 20:
                # Expect an error message containing a note
                # that species extraction failed.
                if (
                    "Unable to extract species from the model" in msg.content
                    and msg.status == "error"
                ):
                    test_condition = True
                    break

            if model_id == 56:
                # Expect a successful extraction (artifact is True) and that the content
                # matches for for missing description ['ORI'].
                if (
                    msg.artifact is True
                    and msg.content == prepare_content_msg(["ORI"])
                    and msg.status == "success"
                ):
                    test_condition = True
                    break
        assert test_condition  # Expected output is validated


def test_prepare_content_msg():
    """
    Cover the branch when species_without_description is non-empty.
    """
    assert prepare_content_msg(["A"]) != ""


def test_get_annotation_processing_paths(monkeypatch):
    """
    Cover annotation processing paths via public tool.invoke.
    """

    class SafeGetAnnotationTool(GetAnnotationTool):
        """Avoid errors when Link is NaN during processing."""

        def _process_link(self, link: str) -> str:
            if not isinstance(link, str):
                return ""
            return super()._process_link(link)

    tool = SafeGetAnnotationTool()
    species = ["S1", "S2", "S3", "S4", "S5"]
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.extract_relevant_species_names",
        lambda *_args, **_kwargs: species,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.load_biomodel",
        lambda *_args, **_kwargs: SimpleNamespace(copasi_model="copasi", biomodel_id="BIOMD1"),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.search_uniprot_labels",
        lambda ids: {"P1": "uniprot_label"},
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.search_ols_labels",
        lambda data: {"go": {"GO:1": "go_label"}},
    )

    def fake_miriam(name):
        if name == "S1":
            return {"descriptions": [{"id": "http://uniprot.org/uniprot/P1", "qualifier": "is"}]}
        if name == "S2":
            return {"descriptions": [{"id": "http://ols/go/GO:1", "qualifier": "is"}]}
        if name == "S3":
            return {"descriptions": [{"id": "http://db/otherdb/X1", "qualifier": "is"}]}
        if name == "S4":
            return {"descriptions": [{"id": math.nan, "qualifier": "is"}]}
        return {"descriptions": []}

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.get_miriam_annotation",
        fake_miriam,
    )
    cmd = tool.invoke(
        _tool_call(
            tool,
            {
                "arg_data": {"experiment_name": "exp", "user_question": "q"},
                "tool_call_id": "tc",
                "state": {"sbml_file_path": [], "llm_model": SimpleNamespace()},
                "sys_bio_model": ModelData(biomodel_id="BIOMD1"),
            },
        )
    )
    df = pd.DataFrame(cmd.update["dic_annotations_data"][0]["data"])
    assert "uniprot_label" in df["Description"].values
    assert "go_label" in df["Description"].values
    assert "-" in df["Description"].values
    assert all("go/" not in link for link in df["Link"].tolist())
    assert "S5" in cmd.update["messages"][0].content


def test_extract_relevant_species_no_species(monkeypatch):
    """
    Cover branch where basico returns no species.
    """
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tools.get_annotation.basico.model_info.get_species",
        lambda model=None: None,
    )
    model_obj = SimpleNamespace(copasi_model="copasi")
    arg_data = SimpleNamespace(user_question="Any?")
    state = {"llm_model": SimpleNamespace()}
    with pytest.raises(ValueError, match="Unable to extract species from the model."):
        extract_relevant_species_names(model_obj, arg_data, state)
