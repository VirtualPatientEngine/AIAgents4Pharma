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
    species: bool = Field(description="Get species from the model.")
    parameters: bool = Field(description="Get parameters from the model.")
    compartments: bool = Field(description="Get compartments from the model.")
    units: bool = Field(description="Get units from the model.")
    description: bool = Field(description="Get description from the model.")
    name: bool = Field(description="Get name from the model.")

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
    name: str = "get_parameters"
    description: str = "A tool for extracting model information."
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
            dic_results['Species'] = df_species.index.tolist()
            dic_results['Species'] = ','.join(dic_results['Species'])

        # Extract parameters from the model
        if requested_model_info.parameters:
            df_parameters = basico.model_info.get_parameters(model=model_obj.copasi_model)
            dic_results['Parameters'] = df_parameters.index.tolist()
            dic_results['Parameters'] = ','.join(dic_results['Parameters'])

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