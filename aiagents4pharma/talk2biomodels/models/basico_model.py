#!/usr/bin/env python3

"""
BasicoModel class for loading and simulating SBML models
using the basico package.
"""

import logging
from typing import Optional, Dict, Union
from pydantic import Field, model_validator
import pandas as pd
import basico
from .sys_bio_model import SysBioModel

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicoModel(SysBioModel):
    """
    Model that loads and simulates SBML models using the basico package.
    Can load models from an SBML file or download them using a BioModels biomodel_id.
    """
    biomodel_id: Optional[Union[int, str]] = Field(None,
                                description="BioModels model ID to download and load")
    sbml_file_path: Optional[str] = Field(None, description="Path to an SBML file to load")
    simulation_results: Optional[str] = None
    name: Optional[str] = Field("", description="Name of the model")
    description: Optional[str] = Field("", description="Description of the model")

    # Additional attribute not included in the schema
    copasi_model: Optional[object] = None  # Holds the loaded Copasi model

    @model_validator(mode="after")
    def check_biomodel_id_or_sbml_file_path(self):
        """
        Validate that either biomodel_id or sbml_file_path is provided.
        """
        if not self.biomodel_id and not self.sbml_file_path:
            logger.error("Either biomodel_id or sbml_file_path must be provided.")
            raise ValueError("Either biomodel_id or sbml_file_path must be provided.")
        if self.biomodel_id:
            self.copasi_model = basico.load_biomodel(self.biomodel_id)
            self.description = basico.biomodels.get_model_info(self.biomodel_id)["description"]
            self.name = basico.model_info.get_model_name(model=self.copasi_model)
        elif self.sbml_file_path:
            self.copasi_model = basico.load_model(self.sbml_file_path)
            self.description = basico.model_info.get_notes(model=self.copasi_model)
            self.name = basico.model_info.get_model_name(model=self.copasi_model)
        return self

    def update_parameters(self, parameters: Dict[str, Union[float, int]]) -> None:
        """
        Update model parameters with new values.
        """
        # Update parameters in the model
        for param_name, param_value in parameters.items():
            # check if the param_name is not None
            if param_name is None:
                continue
            # Extract all parameters and species from the model
            df_all_params = basico.model_info.get_parameters(model=self.copasi_model)
            df_all_species = basico.model_info.get_species(model=self.copasi_model)
            # if param is a kinetic parameter
            if param_name in df_all_params.index.tolist():
                basico.model_info.set_parameters(name=param_name,
                                            exact=True,
                                            initial_value=param_value,
                                            model=self.copasi_model)
            # if param is a species
            elif param_name in df_all_species.index.tolist():
                basico.model_info.set_species(name=param_name,
                                            exact=True,
                                            initial_concentration=param_value,
                                            model=self.copasi_model)
            else:
                logger.error("Parameter/Species %s not found in the model.", param_name)
                raise ValueError(f"Parameter/Species {param_name} not found in the model.")

    def simulate(self, duration: Union[int, float] = 10, interval: int = 10) -> pd.DataFrame:
        """
        Simulate the COPASI model over a specified range of time points.
        
        Args:
            duration: Duration of the simulation in time units.
            interval: Interval between time points in the simulation.
        
        Returns:
            Pandas DataFrame with time-course simulation results.
        """
        # Run the simulation and return results
        df_result = basico.run_time_course(model=self.copasi_model,
                                        intervals=interval,
                                        duration=duration)
        # # Replace curly braces in column headers with square brackets
        # # Because curly braces in the world of LLMS are used for
        # # structured output
        # df_result.columns = df_result.columns.str.replace('{', '[', regex=False).\
        #             str.replace('}', ']', regex=False)
        # Reset the index
        df_result.reset_index(inplace=True)
        # Store the simulation results
        self.simulation_results = df_result
        # Return copy of the simulation results
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
