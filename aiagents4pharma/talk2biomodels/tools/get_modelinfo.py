#!/usr/bin/env python3

"""
Tool for get model information.
"""

import logging
from typing import Type, Optional, Annotated
from dataclasses import dataclass
import basico
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from .load_biomodel import ModelData, load_biomodel

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RequestedModelInfo:
    """
    Dataclass for storing the requested model information.
    """
    species: bool = Field(description="Get species from the model.", default=False)
    parameters: bool = Field(description="Get parameters from the model.", default=False)
    compartments: bool = Field(description="Get compartments from the model.", default=False)
    units: bool = Field(description="Get units from the model.", default=False)
    description: bool = Field(description="Get description from the model.", default=False)
    name: bool = Field(description="Get name from the model.", default=False)

class GetModelInfoInput(BaseModel):
    """
    Input schema for the GetModelInfo tool.
    """
    requested_model_info: RequestedModelInfo = Field(description="requested model information")
    sys_bio_model: ModelData = Field(description="model data")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class GetModelInfoTool(BaseTool):
    """
    This tool ise used extract model information.
    """
    name: str = "get_modelinfo"
    description: str = """A tool for extracting name,
                    description, species, parameters,
                    compartments, and units from a model."""
    args_schema: Type[BaseModel] = GetModelInfoInput

    def _run(self,
            requested_model_info: RequestedModelInfo,
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict, InjectedState],
            sys_bio_model: Optional[ModelData] = None,
             ) -> Command:
        """
        Run the tool.

        Args:
            requested_model_info (RequestedModelInfo): The requested model information.
            tool_call_id (str): The tool call ID. This is injected by the system.
            state (dict): The state of the tool.
            sys_bio_model (ModelData): The model data.

        Returns:
            Command: The updated state of the tool.
        """
        logger.log(logging.INFO,
                   "Calling get_modelinfo tool %s, %s",
                     sys_bio_model,
                   requested_model_info)
        # print (state, 'state')
        sbml_file_path = state['sbml_file_path'][-1] if len(state['sbml_file_path']) > 0 else None
        model_obj = load_biomodel(sys_bio_model,
                                  sbml_file_path=sbml_file_path)
        dic_results = {}
        # Extract species from the model
        if requested_model_info.species:
            df_species = basico.model_info.get_species(model=model_obj.copasi_model)
            if df_species is None:
                raise ValueError("Unable to extract species from the model.")
            # Convert index into a column
            df_species.reset_index(inplace=True)
            dic_results['Species'] = df_species[
                                        ['name',
                                         'compartment',
                                         'type',
                                         'unit',
                                         'initial_concentration',
                                         'display_name']]
            # Convert this into a dictionary
            dic_results['Species'] = dic_results['Species'].to_dict(orient='records')

        # Extract parameters from the model
        if requested_model_info.parameters:
            df_parameters = basico.model_info.get_parameters(model=model_obj.copasi_model)
            if df_parameters is None:
                raise ValueError("Unable to extract parameters from the model.")
            # Convert index into a column
            df_parameters.reset_index(inplace=True)
            dic_results['Parameters'] = df_parameters[
                                        ['name',
                                         'type',
                                         'unit',
                                         'initial_value',
                                         'display_name']]
            # Convert this into a dictionary
            dic_results['Parameters'] = dic_results['Parameters'].to_dict(orient='records')

        # Extract compartments from the model
        if requested_model_info.compartments:
            df_compartments = basico.model_info.get_compartments(model=model_obj.copasi_model)
            dic_results['Compartments'] = df_compartments.index.tolist()
            dic_results['Compartments'] = ','.join(dic_results['Compartments'])

        # Extract description from the model
        if requested_model_info.description:
            dic_results['Description'] = model_obj.description

        # Extract description from the model
        if requested_model_info.name:
            dic_results['Name'] = model_obj.name

        # Extract time unit from the model
        if requested_model_info.units:
            dic_results['Units'] = basico.model_info.get_model_units(model=model_obj.copasi_model)

        # Prepare the dictionary of updated state for the model
        dic_updated_state_for_model = {}
        for key, value in {
                        "model_id": [sys_bio_model.biomodel_id],
                        "sbml_file_path": [sbml_file_path],
                        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        return Command(
            update=dic_updated_state_for_model|{
                    # update the message history
                    "messages": [
                        ToolMessage(
                            content=dic_results,
                            tool_call_id=tool_call_id
                            )
                        ],
                    }
            )
