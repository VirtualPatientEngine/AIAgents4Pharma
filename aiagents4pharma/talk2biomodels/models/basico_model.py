#!/usr/bin/env python3

"""
BasicoModel class for loading and simulating SBML models
using the basico package.
"""

from typing import Optional, Dict, Union
from time import sleep
from urllib.error import URLError
from pydantic import Field, model_validator
import pandas as pd
import basico
from .sys_bio_model import SysBioModel

class BasicoModel(SysBioModel):
    """
    Model that loads and simulates SBML models using the basico package.
    Can load models from an SBML file or download them using a BioModels model_id.
    """
    model_id: Optional[int] = Field(None, description="BioModels model ID to download and load")
    sbml_file_path: Optional[str] = Field(None, description="Path to an SBML file to load")
    simulation_results: Optional[str] = None
    name: Optional[str] = Field("", description="Name of the model")
    description: Optional[str] = Field("", description="Description of the model")

    # Additional attribute not included in the schema
    copasi_model: Optional[object] = None  # Holds the loaded Copasi model

    @model_validator(mode="after")
    def check_model_id_or_sbml_file_path(self):
        """
        Validate that either model_id or sbml_file_path is provided.
        """
        if not self.model_id and not self.sbml_file_path:
            raise ValueError("Either model_id or sbml_file_path must be provided.")
        if self.model_id:
            attempts = 0
            max_retries = 5
            while attempts < max_retries:
                try:
                    self.copasi_model = basico.load_biomodel(self.model_id)
                    break
                except URLError as e:
                    attempts += 1
                    sleep(10*attempts)
                    if attempts >= max_retries:
                        raise e
            self.description = basico.biomodels.get_model_info(self.model_id)["description"]
        elif self.sbml_file_path:
            self.copasi_model = basico.load_model(self.sbml_file_path)
        return self

    def simulate(self,
                 parameters: Optional[Dict[str, Union[float, int]]] = None,
                 duration: Union[int, float] = 10,
                 interval: int = 10
                 ) -> pd.DataFrame:
        """
        Simulate the COPASI model over a specified range of time points.
        
        Args:
            parameters: Dictionary of model parameters to update before simulation.
            duration: Duration of the simulation in time units.
            interval: Interval between time points in the simulation.
        
        Returns:
            Pandas DataFrame with time-course simulation results.
        """

        # Update parameters in the model
        if parameters:
            for param_name, param_value in parameters.items():
                # check if the param_name is not None
                if param_name is None:
                    continue
                # if param is a kinectic parameter
                df_all_params = basico.model_info.get_parameters(model=self.copasi_model)
                if param_name in df_all_params.index.tolist():
                    basico.model_info.set_parameters(name=param_name,
                                                exact=True,
                                                initial_value=param_value,
                                                model=self.copasi_model)
                # if param is a species
                else:
                    basico.model_info.set_species(name=param_name,
                                                exact=True,
                                                initial_concentration=param_value,
                                                model=self.copasi_model)

        # Run the simulation and return results
        df_result = basico.run_time_course(model=self.copasi_model,
                                        intervals=interval,
                                        duration=duration)
        df_result.columns = df_result.columns.str.replace('{', '[', regex=False).\
                    str.replace('}', ']', regex=False)
        df_result.reset_index(inplace=True)
        self.simulation_results = df_result
        return df_result.copy()

    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        """
        Retrieve metadata specific to the COPASI model.
        
        Returns:
            Dictionary of model metadata.
        """
        return {
            "Model Type": "SBML Model (COPASI)",
            "Parameter Count": len(basico.get_parameters())
        }
