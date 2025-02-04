#!/usr/bin/env python3

"""
Tool for parameter scan.
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
    species_name: List[str] = Field(description="species name", default=[])
    species_concentration: List[Union[int, float]] = Field(
        description="initial species concentration",
        default=[])

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
class ReocurringData:
    """
    Dataclass for species that reoccur. In other words, the concentration
    of the species resets to a certain value after a certain time interval.
    """
    data: List[TimeSpeciesNameConcentration] = Field(
        description="time, name, and concentration data of species that reoccur",
        default=[])

@dataclass
class ArgumentData:
    """
    Dataclass for storing the argument data.
    """
    time_data: TimeData = Field(description="time data", default=None)
    species_data: SpeciesData = Field(
        description="species name and initial concentration data")
    reocurring_data: ReocurringData = Field(
        description="""Concentration and time data of species that reoccur
            For example, a species whose concentration resets to a certain value
            after a certain time interval""")
    steadystate_name: str = Field(
        description="""An AI assigned `_` separated name of
        the steady state experiment based on human query""")

def add_rec_events(model_object, reocurring_data):
    """
    Add reocurring events to the model.
    """
    for row in reocurring_data.data:
        tp, sn, sc = row.time, row.species_name, row.species_concentration
        basico.add_event(f'{sn}_{tp}',
                            f'Time > {tp}',
                            [[sn, str(sc)]],
                            model=model_object.copasi_model)

def run_steady_state(model_object, dic_species_data):
    """
    Run the steady state analysis.

    Args:
        model_object: The model object.
        dic_species_data: Dictionary of species data.

    Returns:
        DataFrame: The results of the steady state analysis.
    """
    # Update the fixed model species and parameters
    # These are the initial conditions of the model
    # set by the user
    model_object.update_parameters(dic_species_data)
    logger.log(logging.INFO, "Running steady state analysis")
    # Run the steady state analysis
    output = basico.task_steadystate.run_steadystate(model=model_object.copasi_model)
    if output == 0:
        logger.error("Steady state analysis failed")
        raise ValueError("A steady state was not found")
    logger.log(logging.INFO, "Steady state analysis successful")
    # Store the steady state results in a DataFrame
    df_steady_state = basico.model_info.get_species(model=model_object.copasi_model)
    logger.log(logging.INFO, "Steady state results with shape %s", df_steady_state.shape)
    return df_steady_state

class SteadyStateInput(BaseModel):
    """
    Input schema for the steady state tool.
    """
    sys_bio_model: ModelData = Field(description="model data",
                                     default=None)
    arg_data: ArgumentData = Field(description=
                                   """time, species, and reocurring data
                                   as well as the steady state data""",
                                   default=None)
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SteadyStateTool(BaseTool):
    """
    Tool for steady state analysis.
    """
    name: str = "steady_state"
    description: str = """A tool to simulate a model and perform
                        steady state analysisto answer questions
                        about the steady state of species."""
    args_schema: Type[BaseModel] = SteadyStateInput

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
            Command: The updated state of the tool.
        """
        logger.log(logging.INFO, "Calling steady_state tool %s, %s",
                   sys_bio_model, arg_data)
        # print (f'Calling steady_state tool {sys_bio_model}, {arg_data}, {tool_call_id}')
        sbml_file_path = state['sbml_file_path'][-1] if len(state['sbml_file_path']) > 0 else None
        model_object = load_biomodel(sys_bio_model,
                                  sbml_file_path=sbml_file_path)
        # Prepare the dictionary of species data
        # that will be passed to the simulate method
        # of the BasicoModel class
        dic_species_data = {}
        if arg_data:
            # Prepare the dictionary of species data
            if arg_data.species_data is not None:
                dic_species_data = dict(zip(arg_data.species_data.species_name,
                                            arg_data.species_data.species_concentration))
            # Add reocurring events (if any) to the model
            if arg_data.reocurring_data is not None:
                add_rec_events(model_object, arg_data.reocurring_data)
        # Run the parameter scan
        df_steady_state = run_steady_state(model_object, dic_species_data)
        # Prepare the dictionary of scanned data
        # that will be passed to the state of the graph
        dic_steady_state_data = {
            'name': arg_data.steadystate_name,
            'source': sys_bio_model.biomodel_id if sys_bio_model.biomodel_id else 'upload',
            'tool_call_id': tool_call_id,
            'data': df_steady_state.to_dict(orient='records')
        }
        # Prepare the dictionary of updated state for the model
        dic_updated_state_for_model = {}
        for key, value in {
            "model_id": [sys_bio_model.biomodel_id],
            "sbml_file_path": [sbml_file_path],
            "dic_steady_state_data": [dic_steady_state_data]
            }.items():
            if value:
                dic_updated_state_for_model[key] = value
        # Return the updated state
        return Command(
                update=dic_updated_state_for_model|{
                # Update the message history
                "messages": [
                ToolMessage(
                        content=f'''Steady state analysis of
                                {arg_data.steadystate_name}
                                are ready''',
                        tool_call_id=tool_call_id
                        )
                    ],
                }
            )
