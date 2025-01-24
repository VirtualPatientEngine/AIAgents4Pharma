"""
This module contains the `GetAnnotationTool` for fetching species annotations 
based on the provided model and species names.
"""

from typing import Type, Optional, List
from dataclasses import dataclass
import requests
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import basico
import pandas as pd
from ..models.basico_model import BasicoModel
from aiagents4pharma.talk2biomodels.api.uniprot import search_uniprot_labels
from aiagents4pharma.talk2biomodels.api.ols import search_ols_labels
from aiagents4pharma.talk2biomodels.api.kegg import fetch_kegg_annotations

@dataclass
class ModelData:
    """Dataclass for storing model data."""
    modelid: Optional[int] = None
    sbml_file_path: Optional[str] = None
    model_object: Optional[BasicoModel] = None

class GetAnnotationInput(BaseModel):
    """Input schema for the GetAnnotation tool."""
    species_names: Optional[List[str]] = Field(default=None,
                                               description="List of species names to fetch annotations for.")
    sys_bio_model: ModelData = ModelData()

# def get_protein_name_or_label(identifier: str) -> str:
#     """Determine and fetch protein name or label based on the identifier format."""
#     if identifier.startswith('P0'):  # Assuming UniProt ID starts with 'P'
#         return fetch_from_uniprot(identifier)
#     elif identifier.startswith(("PATO","CHEBI","SBO","FMA")):  # Assuming PATO ID starts with 'PATO'
#         return search_ols_labels(identifier)
#     elif identifier.startswith('C0'):
#         return fetch_kegg_compound_name(identifier)
# #     return "-"
def parse_entries(input_data: str) -> List[dict[str, str]]:
    """
    Parse tab-separated input data into a list of dictionaries.

    Args:
        input_data (str): The tab-separated string input.

    Returns:
        List[Dict[str, str]]: Parsed list of dictionaries with keys 'Id' and 'Database'.
    """
    entries = []
    for line in input_data.strip().split("\n"):
        fields = line.split("\t")
        if len(fields) >= 6:  # Ensure there are enough fields
            entry = {
                "Id": fields[4].strip(),
                "Database": fields[5].strip()
            }
            entries.append(entry)
    return entries
from typing import List, Dict

def get_protein_name_or_label(data: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Determine and fetch protein names or labels based on the database type for multiple entries.

    Args:
        data (List[Dict[str, str]]): A list of dictionaries containing 'Id' and 'Database'.

    Returns:
        Dict[str, str]: A mapping of identifiers to their names or labels.
    """
    results = {}

    # Group data by database type for efficient processing
    grouped_data = {}
    for entry in data:
        identifier = entry.get('Id')
        database = entry.get('Database', '').lower()
        if identifier and database:
            grouped_data.setdefault(database, []).append(identifier)
        else:
            # Handle missing fields
            results[identifier or "unknown"] = "-"

    # Fetch results for each database type
    for database, identifiers in grouped_data.items():
        try:
            if database == 'uniprot':
                # Fetch UniProt data for the batch
                annotations = search_uniprot_labels(identifiers)
                results.update(annotations)
            elif database in {'pato', 'chebi', 'sbo', 'fma', 'pr'}:
                # Fetch OLS annotations for the batch
                annotations = search_ols_labels([{"Id": id_, "Database": database} for id_ in identifiers])
                for id_ in identifiers:
                    results[id_] = annotations.get(database, {}).get(id_, "-")
            else :
                data = [{"Id": identifier, "Database": "kegg.compound"} for identifier in identifiers]
                annotations = fetch_kegg_annotations(data)
                # result = annotations.get("kegg.annotation", {}).get(identifier, "-")
                # results[identifier] = result if result else "-"
                for identifier in identifiers:
                    results[identifier] = annotations.get(database, {}).get(identifier, "-")
        except Exception as e:
            print(f"Error fetching data for database '{database}': {e}")
            for id_ in identifiers:
                results[id_] = "-"

    return results

# def get_protein_name_or_label(data: List[dict[str, str]]) -> dict[str, str]:
#     """
#     Determine and fetch protein name or label based on the database type.

#     Args:
#         data (List[Dict[str, str]]): A list of dictionaries containing 'Id' and 'Database'.

#     Returns:
#         Dict[str, str]: A mapping of identifiers to their names or labels.
#     """
#     results = {}
#     for entry in data:
#         try :
#             identifier = entry.get('Id')
#             database = entry.get('Database', '').lower()

#             if not identifier or not database:
#                 results[identifier] = "-"  # Handle missing fields
#                 continue

#             if database == 'uniprot':
#                 result = fetch_from_uniprot(identifier)
#                 results[identifier] = result if result else "-"
#             elif database in {'ols', 'pato', 'chebi', 'sbo', 'fma', 'pr'}:
#                 annotations = search_ols_labels([{"Id": identifier, "Database": database}])
#                 result = annotations.get(database, {}).get(identifier, "-")
#                 results[identifier] = result if result else "-"
#             # elif database in {'kegg.annotation'}:
#             #     annotations = fetch_kegg_annotations([{"Id": identifier, "Database": database}])
#             #     result = annotations.get("kegg.annotation", {}).get(identifier, "-")
#             #     results[identifier] = result if result else "-"
#             else:
#                 data = [{"Id": identifier, "Database": "kegg.annotation"}]
#                 annotations = fetch_kegg_annotations(data)
#                 result = annotations.get("kegg.annotation", {}).get(identifier, "-")
#                 results[identifier] = result if result else "-"
#         except Exception as e:
#             print(f"Error fetching data for {identifier}: {e}")
#             results[identifier] = "-"

#     return results


class GetAnnotationTool(BaseTool):
    """Tool for fetching species annotations based on the provided annotation type."""
    name: str = "get_annotation"
    description: str = "Fetches species annotations from the model."
    args_schema: Type[BaseModel] = GetAnnotationInput  # Define schema for inputs
    return_direct: bool = True
    st_session_key: Optional[str] = None
    st_session_df: Optional[str] = None

    def _run(self, species_names: Optional[List[str]] = None, sys_bio_model: ModelData = None,
             _run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
        """Run the tool to fetch species annotations."""

        modelid = sys_bio_model.modelid if sys_bio_model else None
        sbml_file_path = sys_bio_model.sbml_file_path if sys_bio_model else None

        # Early return if session key is not in the session state
        if self.st_session_key and self.st_session_key not in st.session_state:
            return f"Session key {self.st_session_key} not found in Streamlit session state."
        # Handle session state logic for model and SBML file path
        if modelid is None and sbml_file_path is None:
            model_object = st.session_state.get(self.st_session_key, None)
            if model_object is None:
                return "Please provide a BioModels ID or an SBML file path for simulation."
            modelid = model_object.model_id
        model_object = BasicoModel(model_id=modelid) if modelid else BasicoModel(sbml_file_path=sbml_file_path)
        st.session_state[self.st_session_key] = model_object

        df_species = basico.model_info.get_species(model=model_object.copasi_model)
        species_not_found = []
        data = []

        st.session_state[self.st_session_df] = None

        # Default to all species if none provided
        if species_names is None:
            species_names = df_species.index.tolist()

        for species in species_names:
            annotation = basico.get_miriam_annotation(name=species)
            if annotation is None:
                species_not_found.append(species)
                continue

            descriptions = annotation.get("descriptions", [])
            for desc in descriptions:
                data.append({
                    'Species Name': species,
                    'Link': desc['id'],
                    'Qualifier': desc['qualifier']
                })

        annotations_df = pd.DataFrame(data)

        if annotations_df.empty:
            species_msg = (
            f"None of the species entered were found. The following species do not exist: "
            f"{', '.join(species_not_found)}." if species_not_found else
            "No annotations found for the species entered."
            )
            return species_msg

        annotations_df['Id'] = annotations_df['Link'].str.split('/').str[-1]
        annotations_df['Database'] = annotations_df['Link'].str.split('/').str[-2]
        # print(annotations_df)
        # annotations_df['Description'] = annotations_df['Id'].apply(get_protein_name_or_label())
# Convert the results back to a DataFrame if needed

        identifiers = annotations_df[['Id', 'Database']].to_dict(orient='records')
        print(identifiers)
        # identifiers = annotations_df['Id'].tolist()
        descriptions = get_protein_name_or_label(identifiers)

        if descriptions is None: 
            descriptions = {}

        # Add descriptions to the DataFrame
        annotations_df['Description'] = annotations_df['Id'].apply(lambda x: descriptions.get(x, '-'))

        annotations_df.index = annotations_df.index + 1


        st.session_state[self.st_session_df] = annotations_df

        return (
        "All the requested annotations extracted successfully." if not species_not_found else
        f"The following species do not exist, and hence their annotations were not extracted: "
        f"{', '.join(species_not_found)}."
        f"Remaining species annotations is in following table"
         )
