#!/usr/bin/env python3

"""
Tool for plotting a custom figure.
"""

import logging
from typing import Type, Annotated, List, Tuple, Union, Literal
from pydantic import BaseModel, Field
import pandas as pd
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPlotterInput(BaseModel):
    """
    Input schema for the PlotImage tool.
    """
    question: str = Field(description="Description of the plot")
    simulation_name: str = Field(description="Name assigned to the simulation")
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints.
# BaseTool is a Pydantic class and not having type hints
# can lead to unexpected behavior.
# Note: It's important that every field has type hints.
# BaseTool is a Pydantic class and not having type hints
# can lead to unexpected behavior.
class CustomPlotterTool(BaseTool):
    """
    Tool for making custom plots
    """
    name: str = "custom_plotter"
    description: str = "A tool to make custom plots of the simulation results"
    args_schema: Type[BaseModel] = CustomPlotterInput
    response_format: str = "content_and_artifact"

    def _run(self,
             question: str,
             simulation_name: str,
             state: Annotated[dict, InjectedState]
             ) -> Tuple[str, Union[None, List[str]]]:
        """
        Run the tool.

        Args:
            question (str): The question about the custom plot.
            state (dict): The state of the graph.

        Returns:
            str: The answer to the question
        """
        logger.log(logging.INFO, "Calling custom_plotter tool %s", question)
        dic_simulated_data = {}
        for data in state["dic_simulated_data"]:
            for key in data:
                if key not in dic_simulated_data:
                    dic_simulated_data[key] = []
                dic_simulated_data[key] += [data[key]]
        # Create a pandas dataframe from the dictionary
        df = pd.DataFrame.from_dict(dic_simulated_data)
        # Get the simulated data for the current tool call
        df = pd.DataFrame(
                df[df['name'] == simulation_name]['data'].iloc[0]
                )
        # df = pd.DataFrame.from_dict(state['dic_simulated_data'])
        species_names = df.columns.tolist()
        # Exclude the time column
        species_names.remove('Time')
        logging.log(logging.INFO, "Species names: %s", species_names)
        # In the following code, we extract the species
        # from the user question. We use Literal to restrict
        # the species names to the ones available in the
        # simulation results.
        class CustomHeader(BaseModel):
            """
            A list of species based on user question.

            This is a Pydantic model that restricts the species
            names to the ones available in the simulation results.
            
            If no species is relevant, set the attribute
            `relevant_species` to None.
            """
            relevant_species: Union[None, List[Literal[*species_names]]] = Field(
                    description="This is a list of species based on the user question."
                    "It is restricted to the species available in the simulation results."
                    "If no species is relevant, set this attribute to None."
                    "If the user asks for very specific species (for example, using the"
                    "keyword `only` in the question), set this attribute to correspond "
                    "to the species available in the simulation results, otherwise set it to None."
                    )
        # Create an instance of the LLM model
        logging.log(logging.INFO, "LLM model: %s", state['llm_model'])
        llm = state['llm_model']
        llm_with_structured_output = llm.with_structured_output(CustomHeader)
        results = llm_with_structured_output.invoke(question)
        if results.relevant_species is None:
            raise ValueError("No species found in the simulation results \
                             that matches the user prompt.")
        extracted_species = []
        # Extract the species from the results
        # that are available in the simulation results
        for species in results.relevant_species:
            if species in species_names:
                extracted_species.append(species)
        logging.info("Extracted species: %s", extracted_species)
        # Include the time column
        extracted_species.insert(0, 'Time')
        return f"Custom plot {simulation_name}", df[extracted_species].to_dict(orient='records')
