#!/usr/bin/env python3

"""
Tool for simulating a model.
"""

import logging
from dataclasses import dataclass
from typing import Type, Union, List, Annotated
import basico
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from .load_biomodel import ModelData, load_biomodel

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeData:
    """
    Dataclass for storing the time data.
    """
    duration: Union[int, float] = 100
    interval: Union[int, float] = 10

@dataclass
class SpeciesData:
    """
    Dataclass for storing the species data.
    """
    species_name: List[str] = Field(description="species name", default=None)
    species_concentration: List[Union[int, float]] = Field(
        description="initial species concentration",
        default=None)

@dataclass
class TimeSpeciesNameConcentration:
    """
    Dataclass for storing the time, species name, and concentration data.
    """
    time: Union[int, float] = Field(description="time point where the event occurs")
    species_name: str = Field(description="species name")
    species_concentration: Union[int, float] = Field(
        description="species concentration at the time point")

@dataclass
class RecurringData:
    """
    Dataclass for storing the species and time data 
    on reocurring basis.
    """
    data: List[TimeSpeciesNameConcentration] = Field(
        description="species and time data on reocurring basis",
        default=None)

@dataclass
class ArgumentData:
    """
    Dataclass for storing the argument data.
    """
    time_data: TimeData = Field(description="time data", default=None)
    species_data: SpeciesData = Field(
        description="species name and initial concentration data",
        default=None)
    recurring_data: RecurringData = Field(
        description="species and time data on reocurring basis",
        default=None)
    simulation_name: str = Field(
        description="""An AI assigned `_` separated name of
        the simulation based on human query""")

def add_rec_events(model_object, recurring_data):
    """
    Add reocurring events to the model.
    """
    for row in recurring_data.data:
        tp, sn, sc = row.time, row.species_name, row.species_concentration
        basico.add_event(f'{sn}_{tp}',
                            f'Time > {tp}',
                            [[sn, str(sc)]],
                            model=model_object.copasi_model)

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
        dic_species_data = {}
        if arg_data:
            # Prepare the dictionary of species data
            if arg_data.species_data is not None:
                dic_species_data = dict(zip(arg_data.species_data.species_name,
                                            arg_data.species_data.species_concentration))
            # Add recurring events (if any) to the model
            if arg_data.recurring_data is not None:
                add_rec_events(model_object, arg_data.recurring_data)
            # Set the duration and interval
            if arg_data.time_data is not None:
                duration = arg_data.time_data.duration
                interval = arg_data.time_data.interval
        # Update the model parameters
        model_object.update_parameters(dic_species_data)
        logger.log(logging.INFO,
                   "Following species/parameters updated in the model %s",
                   dic_species_data)
        # Simulate the model
        df = model_object.simulate(duration=duration, interval=interval)
        logger.log(logging.INFO, "Simulation results ready with shape %s", df.shape)
        dic_simulated_data = {
            'name': arg_data.simulation_name,
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
                # update the state keys
                # "dic_simulated_data": df.to_dict(),
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"Simulation results of {arg_data.simulation_name}",
                        tool_call_id=tool_call_id
                        )
                    ],
                }
            )
