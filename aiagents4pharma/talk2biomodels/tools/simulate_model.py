#!/usr/bin/env python3

"""
Tool for simulating a model.
"""

from typing import Type, Union, List, Optional
from dataclasses import dataclass
import basico
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import streamlit as st
# import plotly.express as px
from ..models.basico_model import BasicoModel

@dataclass
class ModelData:
    """
    Dataclass for storing the model data.
    """
    modelid: Optional[int] = None
    sbml_file_path: Optional[str] = None

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
    species_name: List[str] = None
    species_concentration: List[Union[int, float]] = None

@dataclass
class TimeSpeciesNameConcentration:
    """
    Dataclass for storing the time, species name, and concentration data.
    """
    time: Union[int, float] = None
    species_name: str = None
    species_concentration: Union[int, float] = None

@dataclass
class RecurringData:
    """
    Dataclass for storing the species and time data 
    on recurring basis.
    """
    data: List[TimeSpeciesNameConcentration] = None

class SimulateModelInput(BaseModel):
    """
    Input schema for the SimulateModel tool.
    """
    model_data: ModelData = Field(description="model data", default=None)
    time_data: TimeData = Field(description="time data", default=None)
    species_data: SpeciesData = Field(description="species data", default=None)
    recurring_data: RecurringData = Field(description="recurring data", default=None)

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SimulateModelTool(BaseTool):
    """
    Tool for simulating a model.
    """
    name: str = "simulate_model"
    description: str = "A tool to simulate a model."
    args_schema: Type[BaseModel] = SimulateModelInput
    st_session_key: str = None

    def _run(self,
                model_data: ModelData = None,
                time_data: TimeData = None,
                species_data: SpeciesData = None,
                recurring_data: RecurringData = None):
        """
        Run the tool.

        Args:
            model_data (Optional[ModelData]): The model data.
            time_data (Optional[TimeData]): The time data.
            species_data (Optional[SpeciesData]): The species data.
            recurring_data (Optional[RecurringData]): The recurring data.

        Returns:
            str: The result of the simulation.
        """
        # st_session_key = self.st_session_key
        # Retrieve the model ID, duration, and interval
        modelid = model_data.modelid if model_data is not None else None
        # duration = time_data.duration if time_data is not None else 100.0
        # interval = time_data.interval if time_data is not None else 10
        # Prepare the dictionary of species data
        # that will be passed to the simulate method
        # of the BasicoModel class
        # dic_species_data = None
        # if species_data is not None:
        #     dic_species_data = dict(zip(species_data.species_name,
        #                             species_data.species_concentration))
        dic_species_data = dict(zip(species_data.species_name, \
            species_data.species_concentration)) \
                if species_data is not None else None
        # Retrieve the SBML file path from the Streamlit session state
        # otherwise retrieve it from the model_data object if the user
        # has provided it.
        sbml_file_path = model_data.sbml_file_path if model_data is not None else None
        if self.st_session_key:
            if self.st_session_key not in st.session_state:
                return f"Session key {self.st_session_key} not found in Streamlit session state."
            if 'sbml_file_path' in st.session_state:
                sbml_file_path = st.session_state.sbml_file_path
        # Check if both modelid and sbml_file_path are None
        if modelid is None and sbml_file_path is None:
            # Then load the model from the Streamlit session state
            # if the streamlit session exists
            if self.st_session_key:
                model_object = st.session_state[self.st_session_key]
                # If this model object is None, then return an error message
                if model_object is None:
                    return "Please provide a BioModels ID or an SBML file path for simulation."
                # Retrieve the model ID from the model object
                modelid = model_object.model_id
            else:
                # Otherwise return an error message
                return "Please provide a BioModels ID or an SBML file path for simulation."
        elif modelid:
            # Create a BasicoModel object with the model ID
            # model_object = BasicoModel(model_id=modelid)
            model_object = BasicoModel(model_id=modelid)
            # Save the model object in the Streamlit session state
            st.session_state[self.st_session_key] = model_object
        elif sbml_file_path:
            # Create a BasicoModel object with the SBML file path
            model_object = BasicoModel(sbml_file_path=sbml_file_path)
            modelid = model_object.model_id
            # Save the model object in the Streamlit session state
            st.session_state[self.st_session_key] = model_object
        # Add recurring events (if any) to the model
        if recurring_data is not None:
            for row in recurring_data.data:
                tp, sn, sc = row.time, row.species_name, row.species_concentration
                basico.add_event(f'{sn}_{tp}',
                                 f'Time > {tp}',
                                 [[sn, str(sc)]],
                                 model=model_object.copasi_model)
                # print (f'Added event {sn}_{tp} at time {tp} \
                #        for species {sn} with concentration {sc}')
        # Simulate the model
        df = model_object.simulate(
            parameters=dic_species_data,
            duration=time_data.duration if time_data is not None else 100.0,
            interval=time_data.interval if time_data is not None else 10)
        # Convert the DataFrame to long format for plotting
        # and ignore the index column
        df = df.melt(id_vars='Time',
                    var_name='Species',
                    value_name='Concentration')
        # Plot the simulation results using Plotly
        # fig = px.line(df,
        #                 x='Time',
        #                 y='Concentration',
        #                 color='Species',
        #                 title=f"Concentration of Species over Time in the model {modelid}",
        #                 height=600,
        #                 width=800
        # )
        # Display the plot in Streamlit
        # st.plotly_chart(fig, use_container_width = False)
        # if modelid is None:
        #     modelid = "internal"
        # return f"Simulation results for the model {modelid}."
        return "Simulation results for the model."

    def get_metadata(self):
        """
        Get metadata for the tool.

        Returns:
            dict: The metadata for the tool.
        """
        return {
            "name": self.name,
            "description": self.description
        }
