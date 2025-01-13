#!/usr/bin/env python3

"""
Tool for fetching species and parameters from the model.
"""

from urllib.error import URLError
from time import sleep
from typing import Type
import basico
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import streamlit as st

class FetchParametersInput(BaseModel):
    """
    Input schema for the ResetModel tool.
    """
    fetch_species: bool = Field(description="Fetch species from the model.")
    fetch_parameters: bool = Field(description="Fetch parameters from the model.")

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class FetchParametersTool(BaseTool):
    """
    This tool fetches species and parameters from the model 
    and returns them as a string in a dictionary.
    """
    name: str = "fetch_parameters"
    description: str = "A tool for fetching species and parameters from the model."
    args_schema: Type[BaseModel] = FetchParametersInput
    st_session_key: str = None

    def _run(self,
             fetch_species: bool,
             fetch_parameters: bool
             ) -> str:
        """
        Run the tool.

        Args:
            fetch_species (bool): Fetch species from the model.
            fetch_parameters (bool): Fetch parameters from the model.

        Returns:
            dict: The species and parameters from the model.
        """
        model_obj = st.session_state[self.st_session_key]
        # Extract species from the model
        species = []
        if fetch_species:
            df_species = self.get_species_and_parameters(model_obj, fetch_species=True)
            species = df_species.index.tolist()
            species = ','.join(species)

        # Extract parameters from the model
        parameters = []
        if fetch_parameters:
            df_parameters = self.get_species_and_parameters(model_obj, fetch_species=False)
            parameters = df_parameters.index.tolist()
            parameters = ','.join(parameters)
        return {'Species': species, 'Parameters': parameters}

    def get_species_and_parameters(self,
                                   model_obj,
                                   fetch_species: bool):
        """
        Get the species from the model.
        """
        attempts = 0
        max_retries = 3
        while attempts < max_retries:
            try:
                if fetch_species:
                    df = basico.model_info.get_species(model=model_obj.copasi_model)
                else:
                    df = basico.model_info.get_parameters(model=model_obj.copasi_model)
                break
            except URLError as e:
                attempts += 1
                sleep(10)
                if attempts >= max_retries:
                    raise e
        return df
