#!/usr/bin/env python3

"""
Tool for simulating a model.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from .load_biomodel import ModelData, load_biomodel
from .load_arguments import ArgumentData, add_rec_events
from .utils import get_model_units

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulateModelInput(BaseModel):
    """
    Input schema for the SimulateModel tool.
    """
    sys_bio_model: ModelData = Field(description="model data",
                                     default=None)
    arg_data: ArgumentData = Field(description=
                                   """time, species, and reocurring data
                                   as well as the simulation name""",
                                   default=None)
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SimulateModelTool(BaseTool):
    """
    Tool for simulating a model.
    """
    name: str = "simulate_model"
    description: str = "A tool to simulate a biomodel"
    args_schema: Type[BaseModel] = SimulateModelInput

    def _run(self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        sys_bio_model: ModelData = None,
        arg_data: ArgumentData = None
    ) -> Command:
        """
        Run the tool.

        Args:
            tool_call_id (str): The tool call ID. This is injected by the system.
            state (dict): The state of the tool.
            sys_bio_model (ModelData): The model data.
            arg_data (ArgumentData): The argument data.

        Returns:
            str: The result of the simulation.
        """
        logger.log(logging.INFO,
                   "Calling simulate_model tool %s, %s",
                   sys_bio_model,
                   arg_data)
        sbml_file_path = state['sbml_file_path'][-1] if len(state['sbml_file_path']) > 0 else None
        model_object = load_biomodel(sys_bio_model,
                                  sbml_file_path=sbml_file_path)
        # Prepare the dictionary of species data
        # that will be passed to the simulate method
        # of the BasicoModel class
        duration = 100.0
        interval = 10
        dic_species_to_be_analyzed_before_experiment = {}
        if arg_data:
            # Prepare the dictionary of species data
            if arg_data.species_to_be_analyzed_before_experiment is not None:
                dic_species_to_be_analyzed_before_experiment = dict(
                    zip(arg_data.species_to_be_analyzed_before_experiment.species_name,
                        arg_data.species_to_be_analyzed_before_experiment.species_concentration))
            # Add reocurring events (if any) to the model
            if arg_data.reocurring_data is not None:
                add_rec_events(model_object, arg_data.reocurring_data)
            # Set the duration and interval
            if arg_data.time_data is not None:
                duration = arg_data.time_data.duration
                interval = arg_data.time_data.interval
        # Update the model parameters
        model_object.update_parameters(dic_species_to_be_analyzed_before_experiment)
        logger.log(logging.INFO,
                   "Following species/parameters updated in the model %s",
                   dic_species_to_be_analyzed_before_experiment)
        # Simulate the model
        df = model_object.simulate(duration=duration, interval=interval)
        logger.log(logging.INFO, "Simulation results ready with shape %s", df.shape)
        dic_simulated_data = {
            'name': arg_data.experiment_name,
            'source': sys_bio_model.biomodel_id if sys_bio_model.biomodel_id else 'upload',
            'tool_call_id': tool_call_id,
            'data': df.to_dict()
        }
        # Prepare the dictionary of updated state
        dic_updated_state_for_model = {}
        for key, value in {
            "model_id": [sys_bio_model.biomodel_id],
            "sbml_file_path": [sbml_file_path],
            "dic_simulated_data": [dic_simulated_data],
            }.items():
            if value:
                dic_updated_state_for_model[key] = value
        # Return the updated state of the tool
        return Command(
                update=dic_updated_state_for_model|{
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"Simulation results of {arg_data.experiment_name}",
                        tool_call_id=tool_call_id,
                        artifact=get_model_units(model_object)
                        )
                    ],
                }
            )
