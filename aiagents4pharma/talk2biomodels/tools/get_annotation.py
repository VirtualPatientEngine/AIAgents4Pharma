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

def get_protein_name_or_label(identifier: str) -> Optional[str]:
    """Fetch the protein name or label based on the identifier."""
    if identifier.startswith('P0'):  # Assuming Uniprot ID starts with 'P'
        url = f"https://www.uniprot.org/uniprot/{identifier}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Name not found')
        except requests.exceptions.RequestException:
            return "Error: Unable to fetch data from Uniprot."
    elif identifier.startswith("PATO"):  # Assuming PATO ID starts with 'PATO'
        formatted_pato_id = identifier.replace(":", "_")
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/pato/terms?iri=http://purl.obolibrary.org/obo/{formatted_pato_id}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['_embedded']['terms'][0].get('label', 'Label not found')
        except requests.exceptions.RequestException:
            return "Error: Unable to fetch data from OLS."
    return "Error: Unrecognized ID format."

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

        # Retrieve species from model and process annotations
        df_species = basico.model_info.get_species(model=model_object.copasi_model)
        species_not_found = []
        data = []

        st.session_state[self.st_session_df] = None

        # Default to all species if none provided
        if species_names is None:
            species_names = df_species.index().tolist()

        for species in species_names:
            annotation = basico.get_miriam_annotation(name=species)
            if annotation is None:
                species_not_found.append(species)
                continue

            descriptions = annotation.get("descriptions", [])
            for desc in descriptions:
                data.append({
                    'species name': species,
                    'link': desc['id'],
                    'qualifier': desc['qualifier']
                })

        annotations_df = pd.DataFrame(data)

        if annotations_df.empty:
            species_msg = (
            f"None of the species entered were found. The following species do not exist: "
            f"{', '.join(species_not_found)}." if species_not_found else 
            "No annotations found for the species entered."
            )
            return species_msg

        annotations_df['Id'] = annotations_df['link'].str.split('/').str[-1]
        annotations_df['database_name'] = annotations_df['link'].str.split('/').str[-2]
        annotations_df['Suggested Name'] = annotations_df['Id'].apply(get_protein_name_or_label)

        st.session_state[self.st_session_df] = annotations_df

        return (
        "All the requested annotations extracted successfully." if not species_not_found else
        f"The following species do not exist, and hence their annotations were not extracted: "
        f"{', '.join(species_not_found)}."
         )
