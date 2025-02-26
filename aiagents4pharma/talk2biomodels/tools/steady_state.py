#!/usr/bin/env python3

"""
Tool for parameter scan.
"""

import logging
from typing import Type, Annotated
import basico
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from .load_biomodel import ModelData, load_biomodel
from .load_arguments import ArgumentData, add_rec_events

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_steady_state(model_object,
                     dic_species_to_be_analyzed_before_experiment):
    """
    Run the steady state analysis.

    Args:
        model_object: The model object.
        dic_species_to_be_analyzed_before_experiment: Dictionary of species data.

    Returns:
        DataFrame: The results of the steady state analysis.
    """
    # Update the fixed model species and parameters
    # These are the initial conditions of the model
    # set by the user
    model_object.update_parameters(dic_species_to_be_analyzed_before_experiment)
    logger.log(logging.INFO, "Running steady state analysis")
    # Run the steady state analysis
    output = basico.task_steadystate.run_steadystate(model=model_object.copasi_model)
    if output == 0:
        logger.error("Steady state analysis failed")
        raise ValueError("A steady state was not found")
    logger.log(logging.INFO, "Steady state analysis successful")
    # Store the steady state results in a DataFrame
    df_steady_state = basico.model_info.get_species(model=model_object.copasi_model).reset_index()
    # print (df_steady_state)
    # Rename the column name to species_name
    df_steady_state.rename(columns={'name': 'species_name'},
                           inplace=True)
    # Rename the column concentration to steady_state_concentration
    df_steady_state.rename(columns={'concentration': 'steady_state_concentration'},
                           inplace=True)
    # Rename the column transition_time to steady_state_transition_time
    df_steady_state.rename(columns={'transition_time': 'steady_state_transition_time'},
                           inplace=True)
    # Drop some columns
    df_steady_state.drop(columns=
                         [
                            'initial_particle_number',
                            'initial_expression',
                            'expression',
                            'particle_number',
                            'type',
                            'particle_number_rate',
                            'key',
                            'sbml_id',
                            'display_name'],
                            inplace=True)
    logger.log(logging.INFO, "Steady state results with shape %s", df_steady_state.shape)
    return df_steady_state

class SteadyStateInput(BaseModel):
    """
    Input schema for the steady state tool.
    """
    sys_bio_model: ModelData = Field(description="model data",
                                     default=None)
    arg_data: ArgumentData = Field(
        description="time, species, and reocurring data"
                " that must be set before the steady state analysis"
                " as well as the experiment name", default=None)
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SteadyStateTool(BaseTool):
    """
    Tool to bring a model to steady state.
    """
    name: str = "steady_state"
    description: str = "A tool to bring a model to steady state."
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
        logger.log(logging.INFO, "Calling the steady_state tool %s, %s",
                   sys_bio_model, arg_data)
        # print (f'Calling steady_state tool {sys_bio_model}, {arg_data}, {tool_call_id}')
        sbml_file_path = state['sbml_file_path'][-1] if len(state['sbml_file_path']) > 0 else None
        model_object = load_biomodel(sys_bio_model,
                                  sbml_file_path=sbml_file_path)
        # Prepare the dictionary of species data
        # that will be passed to the simulate method
        # of the BasicoModel class
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
        # Run the parameter scan
        df_steady_state = run_steady_state(model_object,
                                           dic_species_to_be_analyzed_before_experiment)
        print (df_steady_state)
        # Prepare the dictionary of scanned data
        # that will be passed to the state of the graph
        dic_steady_state_data = {
            'name': arg_data.experiment_name,
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
                        content=f"Steady state analysis of"
                                f" {arg_data.experiment_name}"
                                " was successful.",
                        tool_call_id=tool_call_id,
                        artifact={'dic_data': df_steady_state.to_dict(orient='records')}
                        )
                    ],
                }
            )
