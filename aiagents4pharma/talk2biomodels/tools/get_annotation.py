#!/usr/bin/env python3

"""
This module contains the `GetAnnotationTool` for fetching species annotations 
based on the provided model and species names.
"""

from typing import List, Optional, Annotated, Type, Union
import logging
from pydantic import BaseModel, Field
import basico
import pandas as pd
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from .load_biomodel import ModelData, load_biomodel
from ..api.uniprot import search_uniprot_labels
from ..api.ols import search_ols_labels
from ..api.kegg import fetch_kegg_annotations

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GetAnnotationInput(BaseModel):
    """
    Input schema for annotation tool.
    """
    sys_bio_model: ModelData = Field(description="model data")
    tool_call_id: Annotated[str, InjectedToolCallId]
    species_names: Union[None, List[str]] = Field(
        default=None,
        description='''List of species names to fetch annotations for.
                      If not provided (i.e. None) annotations for all
                      species in the model will be fetched.'''
    )
    state: Annotated[dict, InjectedState]

class GetAnnotationTool(BaseTool):
    """
    Tool for fetching species annotations based on the provided model and species names.
    """
    name: str = "get_annotation"
    description: str = '''A tool to extract annotations for species names
                        based on the provided model. Annotations include
                        the species name, description, database, ID, link,
                        and qualifier.'''
    args_schema: Type[BaseModel] = GetAnnotationInput
    return_direct: bool = False

    def _run(self,
             species_names: Union[None, List[str]],
             sys_bio_model: ModelData,
             tool_call_id: Annotated[str, InjectedToolCallId],
             state: Annotated[dict, InjectedState]) -> str:
        """
        Run the tool.
        """
        logger.info("Running the GetAnnotationTool tool.")

        # Prepare the model object
        sbml_file_path = state['sbml_file_path'][-1] if state['sbml_file_path'] else None
        model_object = load_biomodel(sys_bio_model, sbml_file_path=sbml_file_path)

        # Extract all the species names from the model
        df_species = basico.model_info.get_species(model=model_object.copasi_model)

        # Fetch annotations for the species names
        species_names = species_names or df_species.index.tolist()
        annotations_df, species_not_found = self._fetch_annotations(species_names)

        # Check if annotations are empty
        # If empty, return a message
        if annotations_df.empty:
            logger.warning("The annotations dataframe is empty.")
            # species_msg = self._generate_empty_result_message(species_not_found)
            # return species_msg
            if species_not_found:
                return f'''The following species do not exist:
                        {", ".join(species_not_found)}.''', False
            return "No annotations found for any of the species entered.", False

        # Process annotations
        annotations_df = self._process_annotations(annotations_df)

        # Update the state with the annotations data
        dic_updated_state_for_model = {}
        for key, value in {
            "model_id": [sys_bio_model.biomodel_id],
            "sbml_file_path": [sbml_file_path],
            "dic_annotations_data": annotations_df.to_dict() if not annotations_df.empty else {}
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        return Command(
            update=dic_updated_state_for_model | {
                "dic_annotations_data": annotations_df.to_dict() 
                                        if not annotations_df.empty else {},
                "messages": [
                    ToolMessage(
                        content=(
                            "Successfully extracted annotations for the species."
                            if not species_not_found else
                            f'''The following species do not exist, and
                            hence their annotations were not extracted:
                            {', '.join(species_not_found)}.'''
                        ),
                        artifact=True,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    def _fetch_annotations(self, species_names: List[str]) -> tuple:
        """
        Fetch annotations for the given species names from the model.
        In this method, we fetch the MIRIAM annotations for the species names.
        If the annotation is not found, we add the species to the list of
        species not found. If the annotation is found, we extract the descriptions
        from the annotation and add them to the data list.

        Args:
            species_names (List[str]): List of species names to fetch annotations for.

        Returns:
            tuple: A tuple containing the annotations dataframe and the species not found list.
        """
        species_not_found = []
        data = []

        # Loop through the species names
        for species in species_names:
            # Get the MIRIAM annotation for the species
            annotation = basico.get_miriam_annotation(name=species)
            # If the annotation is not found, add the species to the list
            if annotation is None:
                species_not_found.append(species)
                continue

            # Extract the descriptions from the annotation
            descriptions = annotation.get("descriptions", [])

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
        return annotations_df, species_not_found

    # def _generate_empty_result_message(self, species_not_found: List[str]) -> str:
    #     """
    #     Generate a message for empty results.
    #     """
    #     if species_not_found:
    #         return f"The following species do not exist: {', '.join(species_not_found)}."
    #     return "No annotations found for any of the species entered."

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
        substrings = ["chebi/", "pato/", "pr/", "fma/", "sbo/"]
        for substring in substrings:
            if substring in link:
                link = link.replace(substring, "")
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

        for entry in data:
            identifier = entry.get('Id')
            database = entry.get('Database', '').lower()
            if identifier and database:
                grouped_data.setdefault(database, []).append(identifier)
            else:
                results[identifier or "unknown"] = "-"

        for database, identifiers in grouped_data.items():
            try:
                if database == 'uniprot':
                    results.update(search_uniprot_labels(identifiers))
                elif database in {'pato', 'chebi', 'sbo', 'fma', 'pr'}:
                    annotations = search_ols_labels([
                        {"Id": id_, "Database": database}
                        for id_ in identifiers
                    ])
                    for id_ in identifiers:
                        results[id_] = annotations.get(database, {}).get(id_, "-")
                else:
                    data = [{"Id": identifier, "Database": "kegg.compound"}
                            for identifier in identifiers]
                    annotations = fetch_kegg_annotations(data)
                    for identifier in identifiers:
                        results[identifier] = annotations.get(database, {}).get(identifier, "-")
            except ImportError as e:
                logger.error("Error fetching data for database '%s': %s", database, e)
                for id_ in identifiers:
                    results[id_] = "-"

        return results
