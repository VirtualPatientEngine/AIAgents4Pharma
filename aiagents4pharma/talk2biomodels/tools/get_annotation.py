
"""
Tool for fetching species annotations from the simulation results.
"""

from typing import Type, Optional, List
from dataclasses import dataclass
import streamlit as st
from pydantic import BaseModel, Field
from langchain.agents.agent_types import AgentType
from langchain_core.tools.base import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from ..models.basico_model import BasicoModel
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
import basico
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import ToolException
from langchain_core.tools import StructuredTool

@dataclass
class ModelData:
    """
    Dataclass for storing the model data.
    """
    modelid: Optional[int] = None
    sbml_file_path: Optional[str] = None
    model_object: Optional[BasicoModel] = None

class GetAnnotationInput(BaseModel):
    """
    Input schema for the GetAnnotation tool.
    """
    species_names: list = Field(default=None, description="List of species names to fetch annotations for.")
    sys_bio_model: ModelData = ModelData()

def _handle_error(error: str) -> str:
    return f"Error: {error}"

class GetAnnotationTool(BaseTool):
    """
    Tool for fetching species annotations based on the provided annotation type.
    """
    name: str = "get_annotation"
    description: str = "Fetches species annotated from the model."
    args_schema: Type[BaseModel] = GetAnnotationInput  # Define schema for inputs
    return_direct: bool = True
    st_session_key: str = None

    def _run(self,
             species_names:  list = None,
             sys_bio_model: ModelData = None,    # Adding species_name as an optional argument
             run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
        """
        Run the tool to fetch species annotations.

        Args:
            species_name (Optional[str]): Specific species name to fetch annotations for.
            sys_bio_model (ModelData): The model data containing model ID, SBML file path, or model object.
            run_manager (Optional[CallbackManagerForToolRun]): The CallbackManagerForToolRun object.

        Returns:
            dict: A dictionary containing the species list.
        """
        st_session_key = self.st_session_key

        modelid = sys_bio_model.modelid if sys_bio_model is not None else None

        sbml_file_path = sys_bio_model.sbml_file_path if sys_bio_model is not None else None
        if st_session_key:
            if st_session_key not in st.session_state:
                return f"Session key {st_session_key} not found in Streamlit session state."
            if 'sbml_file_path' in st.session_state:
                sbml_file_path = st.session_state.sbml_file_path
        # Check if both modelid and sbml_file_path are None
        if modelid is None and sbml_file_path is None:
            # Then load the model from the Streamlit session state
            # if the streamlit session exists
            if st_session_key:
                model_object = st.session_state[st_session_key]
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
            st.session_state[st_session_key] = model_object
        elif sbml_file_path:
            # Create a BasicoModel object with the SBML file path
            model_object = BasicoModel(sbml_file_path=sbml_file_path)
            modelid = model_object.model_id
            # Save the model object in the Streamlit session state
            st.session_state[st_session_key] = model_object

        # try:
        df_species = basico.model_info.get_species(model=model_object.copasi_model)
        data = []
        if species_names is None:
            df_species = basico.model_info.get_species(model=model_object.copasi_model)
            species_names = df_species.index().tolist()      


        for species in species_names:

            if species not in df_species.index:
                error_message = f"The species '{species_names}' you provided was not found in the model data. This could be due to an incorrect species name (e.g., spelling or capitalization errors), the species not being included in the model. Please verify that the species name is correct and exists in the model. If the species is missing, you may need to update the model or check its documentation to ensure the species is included"
                return {"error": error_message}

            annotation = basico.get_miriam_annotation(name=species)
            descriptions = annotation.get("descriptions", [])
            
            for desc in descriptions:
                data.append({
                    'species name': species,
                    'id': desc['id'],
                    'qualifier': desc['qualifier']
                })  
        # Convert to DataFrame for consistent formatting
        annotations_df = pd.DataFrame(data)

        # Create the prompt content for formatting
        prompt_content = f'''
                        Convert the input data into a single table:

                        The table must contain the following columns:
                        - #
                        - Species Name
                        - ID (Clickable)
                        - URI (Clickable)
                        - Qualifier

                        Additional Guidelines:
                        - The column # must contain the row number starting from 1.
                        - Embed the URL for each ID and URI in the table in the markdown format.
                        - Keep the ID and URI columns clickable as it is in df.
                        - Combine all the tables into a single table.
                        - Put all the data in one table.

                        Input:
                        {input}
                        '''

        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt_content),
            ("user", "{input}")]
        )

        # Set up the LLM and output parser
        llm = ChatOpenAI(model="gpt-4o-mini")
        parser = StrOutputParser()
        chain = prompt_template | llm | parser

            # Invoke the chain to format the annotations_df
        return chain.invoke({"input": annotations_df})       

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