#!/usr/bin/env python3

"""
This module contains the `GetAnnotationTool` for fetching species annotations 
based on the provided model and species names.
"""
import math
from typing import List, Annotated, Type, Union, Literal
import logging
from dataclasses import dataclass
import hydra
from pydantic import BaseModel, Field
import basico
import pandas as pd
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
# from langchain_openai import ChatOpenAI
from .load_biomodel import ModelData, load_biomodel
from ..api.uniprot import search_uniprot_labels
from ..api.ols import search_ols_labels
from ..api.kegg import fetch_kegg_annotations

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ols_ontology_abbreviations = {'pato', 'chebi', 'sbo', 'fma', 'pr','go'}

def extract_relevant_species_names(model_object, arg_data, state):
    """
    Extract relevant species names based on the user question.
    """
    # Load hydra configuration
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['tools/get_annotation=default'])
        cfg = cfg.tools.get_annotation
    logger.info("Loaded the following system prompt for the LLM"
                " to get a structured output: %s", cfg.prompt)

    # Extract all the species names from the model
    df_species = basico.model_info.get_species(model=model_object.copasi_model)
    if df_species is None:
        raise ValueError("Unable to extract species from the model.")
    # Get all the species names
    all_species_names = df_species.index.tolist()

    # Define a structured output for the LLM model
    class CustomHeader(BaseModel):
        """
        A list of species based on user question.
        """
        relevant_species: Union[None, List[Literal[*all_species_names]]] = Field(
                description="""List of species based on user question.
                If no relevant species are found, it must be None.""")

    # Create an instance of the LLM model
    llm = state['llm_model']
    # Get the structured output from the LLM model
    llm_with_structured_output = llm.with_structured_output(CustomHeader)
    # Define the question for the LLM model using the prompt
    question = cfg.prompt
    question += f'Here is the user question: {arg_data.user_question}'
    # Invoke the LLM model with the user question
    results = llm_with_structured_output.invoke(question)
    logging.info("Results from the LLM model: %s", results)
    # Check if the returned species names are empty
    if not results.relevant_species:
        raise ValueError("Model does not contain the requested species.")
    extracted_species = []
    # Extract all the species names from the model
    for species in results.relevant_species:
        if species in all_species_names:
            extracted_species.append(species)
    logger.info("Extracted species: %s", extracted_species)
    return extracted_species

def prepare_content_msg(species_without_description: List[str]):
    """
    Prepare the content message.
    """
    content = 'Successfully extracted annotations for the species.'
    if species_without_description:
        content += f'''The descriptions for the following species
                        were not found:
                        {", ".join(species_without_description)}.'''
    return content

@dataclass
class ArgumentData:
    """
    Dataclass for storing the argument data.
    """
    experiment_name: Annotated[str, "An AI assigned _ separated name of"
                                    " the experiment based on human query"
                                    " and the context of the experiment."
                                    " This must be set before the experiment is run."]
    user_question: Annotated[str, "Description of the user question"]

class GetAnnotationInput(BaseModel):
    """
    Input schema for annotation tool.
    """
    arg_data: ArgumentData = Field(description="argument data")
    sys_bio_model: ModelData = Field(description="model data")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

class GetAnnotationTool(BaseTool):
    """
    Tool for fetching species annotations based on the provided model and species names.
    """
    name: str = "get_annotation"
    description: str = '''A tool to extract annotations for a list of species names
                        based on the provided model. Annotations include
                        the species name, description, database, ID, link,
                        and qualifier. The tool can handle multiple species
                        in a single invoke.'''
    args_schema: Type[BaseModel] = GetAnnotationInput
    return_direct: bool = False

    def _run(self,
             arg_data: ArgumentData,
             tool_call_id: Annotated[str, InjectedToolCallId],
             state: Annotated[dict, InjectedState],
             sys_bio_model: ModelData = None) -> str:
        """
        Run the tool.
        """
        logger.info("Running the GetAnnotationTool tool for species %s, %s",
                    arg_data.user_question,
                    arg_data.experiment_name)

        # Prepare the model object
        sbml_file_path = state['sbml_file_path'][-1] if state['sbml_file_path'] else None
        model_object = load_biomodel(sys_bio_model, sbml_file_path=sbml_file_path)

        # Extract relevant species names based on the user question
        list_species_names = extract_relevant_species_names(model_object, arg_data, state)
        print (list_species_names)

        (annotations_df,
         species_without_description) = self._fetch_annotations(list_species_names)

        # Process annotations
        annotations_df = self._process_annotations(annotations_df)

        # Prepare the simulated data
        dic_annotations_data = {
            'name': arg_data.experiment_name,
            'source': sys_bio_model.biomodel_id if sys_bio_model.biomodel_id else 'upload',
            'tool_call_id': tool_call_id,
            'data': annotations_df.to_dict()
        }

        # Update the state with the annotations data
        dic_updated_state_for_model = {}
        for key, value in {
            "model_id": [sys_bio_model.biomodel_id],
            "sbml_file_path": [sbml_file_path],
            "dic_annotations_data": [dic_annotations_data]
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        return Command(
            update=dic_updated_state_for_model | {
                "messages": [
                    ToolMessage(
                        content=prepare_content_msg(species_without_description),
                        artifact=True,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    def _fetch_annotations(self, list_species_names: List[str]) -> tuple:
        """
        Fetch annotations for the given species names from the model.
        In this method, we fetch the MIRIAM annotations for the species names.
        If the annotation is not found, we add the species to the list of
        species not found. If the annotation is found, we extract the descriptions
        from the annotation and add them to the data list.

        Args:
            list_species_names (List[str]): List of species names to fetch annotations for.

        Returns:
            tuple: A tuple containing the annotations dataframe, species not found list,
                   and description not found list.
        """
        description_not_found = []
        data = []

        # Loop through the species names
        for species in list_species_names:
            # Get the MIRIAM annotation for the species
            annotation = basico.get_miriam_annotation(name=species)

            # Extract the descriptions from the annotation
            descriptions = annotation.get("descriptions", [])

            if descriptions == []:
                description_not_found.append(species)
                continue

            # Loop through the descriptions and add them to the data list
            for desc in descriptions:
                data.append({
                    "Species Name": species,
                    "Link": desc["id"],
                    "Qualifier": desc["qualifier"]
                })

        # Create a dataframe from the data list
        annotations_df = pd.DataFrame(data)

        # Return the annotations dataframe and the species not found list
        return annotations_df, description_not_found

    def _process_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process annotations dataframe to add additional information.
        In this method, we add a new column for the ID, a new column for the database,
        and a new column for the description. We then reorder the columns and process
        the link to format it correctly.

        Args:
            annotations_df (pd.DataFrame): Annotations dataframe to process.

        Returns:
            pd.DataFrame: Processed annotations dataframe
        """
        logger.info("Processing annotations.")
        # Add a new column for the ID
        # Get the ID from the link key
        annotations_df['Id'] = annotations_df['Link'].str.split('/').str[-1]

        # Add a new column for the database
        # Get the database from the link key
        annotations_df['Database'] = annotations_df['Link'].str.split('/').str[-2]

        # Fetch descriptions for the IDs based on the database type
        # by qyerying the respective APIs
        identifiers = annotations_df[['Id', 'Database']].to_dict(orient='records')
        descriptions = self._fetch_descriptions(identifiers)

        # Add a new column for the description
        # Get the description from the descriptions dictionary
        # based on the ID. If the description is not found, use '-'
        annotations_df['Description'] = annotations_df['Id'].apply(lambda x:
                                                                   descriptions.get(x, '-'))
        # annotations_df.index = annotations_df.index + 1

        # Reorder the columns
        annotations_df = annotations_df[
            ["Species Name", "Description", "Database", "Id", "Link", "Qualifier"]
        ]

        # Process the link to format it correctly
        annotations_df["Link"] = annotations_df["Link"].apply(self._process_link)

        # Return the processed annotations dataframe
        return annotations_df

    def _process_link(self, link: str) -> str:
        """
        Process link to format it correctly.
        """
        for ols_ontology_abbreviation in ols_ontology_abbreviations:
            if ols_ontology_abbreviation +'/' in link:
                link = link.replace(f"{ols_ontology_abbreviation}/", "")
        if "kegg.compound" in link:
            link = link.replace("kegg.compound/", "kegg.compound:")
        return link

    def _fetch_descriptions(self, data: List[dict[str, str]]) -> dict[str, str]:
        """
        Fetch protein names or labels based on the database type.
        """
        logger.info("Fetching descriptions for the IDs.")
        results = {}
        grouped_data = {}

        # In the following loop, we create a dictionary with database as the key
        # and a list of identifiers as the value. If either the database or the
        # identifier is NaN, we set it to None.
        for entry in data:
            identifier = entry.get('Id')
            database = entry.get('Database')
            # Check if database is NaN
            if isinstance(database, float):
                if math.isnan(database):
                    database = None
                    results[identifier or "unknown"] = "-"
            else:
                database = database.lower()
                grouped_data.setdefault(database, []).append(identifier)

        # In the following loop, we fetch the descriptions for the identifiers
        # based on the database type.
        # Constants

        for database, identifiers in grouped_data.items():
            if database == 'uniprot':
                results.update(search_uniprot_labels(identifiers))
            elif database in ols_ontology_abbreviations:
                annotations = search_ols_labels([
                        {"Id": id_, "Database": database}
                        for id_ in identifiers
                    ])
                for identifier in identifiers:
                    results[identifier] = annotations.get(database, {}).get(identifier, "-")
            elif database == 'kegg.compound':
                data = [{"Id": identifier, "Database": "kegg.compound"}
                        for identifier in identifiers]
                annotations = fetch_kegg_annotations(data)
                for identifier in identifiers:
                    results[identifier] = annotations.get(database, {}).get(identifier, "-")
            else:
                for identifier in identifiers:
                    results[identifier] = "-"
        return results
