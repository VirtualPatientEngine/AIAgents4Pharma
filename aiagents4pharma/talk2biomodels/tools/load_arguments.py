#!/usr/bin/env python3

"""
A utility module for defining the dataclasses
for the arguments to set up initial settings
before the experiment is run.
"""

import logging
from dataclasses import dataclass
from typing import Union, List, Optional, Annotated
from pydantic import Field
import basico

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeData:
    """
    Dataclass for storing the time data.
    """
    duration: Union[int, float] = Field(
        description="Duration of the simulation",
        default=100)
    interval: Union[int, float] = Field(
        description="The interval is the time step or"
        " the step size of the simulation. It is unrelated"
        " to the step size of species concentration and parameter values.",
        default=100)

@dataclass
class SpeciesInitialData:
    """
    Dataclass for storing the species initial data.
    """
    species_name: List[str] = Field(
        description="List of species whose initial concentration is to be set."
        " This does not include species that reoccur or the species whose"
        " concentration is to be determined/observed at the end of the experiment."
        " Do not hallucinate the species name.",
        default=[])
    species_concentration: List[Union[int, float]] = Field(
        description="List of initial concentrations of species."
        " This does not include species that reoccur or the species whose"
        " concentration is to be determined/observed at the end of the experiment."
        " Do not hallucinate the species concentration.",
        default=[])

@dataclass
class TimeSpeciesNameConcentration:
    """
    Dataclass for storing the time,
    species name, and concentration data.
    """
    time: Union[int, float] = Field(description="time point where the event occurs")
    species_name: str = Field(description="species name")
    species_concentration: Union[int, float] = Field(
        description="species concentration at the time point")

@dataclass
class ReocurringData:
    """
    Dataclass for species that reoccur. In other words,
    the concentration of the species resets to a certain
    value after a certain time interval.
    """
    data: List[TimeSpeciesNameConcentration] = Field(
        description="List of time, name, and concentration data"
                    " of species or parameters that reoccur",
                    default=[])

@dataclass
class ArgumentData:
    """
    Dataclass for storing the argument data.
    """
    experiment_name: Annotated[str, "An AI assigned _ separated name of"
                                    " the experiment based on human query"
                                    " and the context of the experiment."
                                    " This must be set before the experiment is run."]
    time_data: Optional[TimeData] = Field(
        description="time data",
        default=None)
    species_to_be_analyzed_before_experiment: Optional[SpeciesInitialData] = Field(
        description="Data of species whose initial concentration"
        " is to be set before the experiment. This does not include"
        " species that reoccur or the species whose concentration"
        " is to be determined at the end of the experiment.",
        default=None)
    reocurring_data: Optional[ReocurringData] = Field(
        description="List of concentration and time data of species that"
        " reoccur. For example, a species whose concentration resets"
        " to a certain value after a certain time interval.",
        default=None)

def add_rec_events(model_object, reocurring_data):
    """
    Add reocurring events to the model.

    Args:
        model_object: The model object.
        reocurring_data: The reocurring data.

    Returns:
        None
    """
    for row in reocurring_data.data:
        tp, sn, sc = row.time, row.species_name, row.species_concentration
        basico.add_event(f'{sn}_{tp}',
                            f'Time > {tp}',
                            [[sn, str(sc)]],
                            model=model_object.copasi_model)
