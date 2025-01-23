#!/usr/bin/env python3

"""
Tool for plotting a custom figure.
"""

import logging
from typing import Type, List, TypedDict, Annotated, Tuple, Union, Literal
from typing import Type, List, TypedDict, Annotated, Tuple, Union, Literal
from pydantic import BaseModel, Field
import pandas as pd
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPlotterInput(BaseModel):
    """
    Input schema for the PlotImage tool.
    """
    question: str = Field(description="Description of the plot")
    state: Annotated[dict, InjectedState]
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
    response_format: str = "content_and_artifact"

    def _run(self,
             question: str,
             state: Annotated[dict, InjectedState]
             ) -> Tuple[str, Union[None, List[str]]]:
        """
        Run the tool.

        Args:
            question (str): The question about the custom plot.
            state (dict): The state of the graph.
            question (str): The question about the custom plot.
            state (dict): The state of the graph.

        Returns:
            str: The answer to the question
        """
        logger.log(logging.INFO, "Calling custom_plotter tool %s", question)
        # Check if the simulation results are available
        # if 'dic_simulated_data' not in state:
        #     return "Please run the simulation first before plotting the figure.", None
        df = pd.DataFrame.from_dict(state['dic_simulated_data'])
        species_names = df.columns.tolist()
        # Exclude the time column
        species_names.remove('Time')
        # In the following code, we extract the species
        # from the user question. We use Literal to restrict
        # the species names to the ones available in the
        # simulation results.
        class CustomHeader(TypedDict):
            """
            A list of species based on user question.
            """
            relevant_species: Union[None, List[Literal[*species_names]]] = Field(
                    description="List of species based on user question. If no relevant species are found, it will be None.")
        # Create an instance of the LLM model
        llm = ChatOpenAI(model=state['llm_model'], temperature=0)
        llm_with_structured_output = llm.with_structured_output(CustomHeader)
        results = llm_with_structured_output.invoke(question)
        extracted_species = []
        # Extract the species from the results
        # that are available in the simulation results
        for species in results['relevant_species']:
            if species in species_names:
                extracted_species.append(species)
        logger.info("Extracted species: %s", extracted_species)
        if len(extracted_species) == 0:
            return "No species found in the simulation results that matches the user prompt.", None
        content = f"Plotted custom figure with species: {', '.join(extracted_species)}"
        return content, extracted_species
