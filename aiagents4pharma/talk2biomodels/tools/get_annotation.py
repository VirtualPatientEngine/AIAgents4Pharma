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
    species_name: Optional[str] = Field(default=None, description="Specific species name to fetch annotations for.")
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
             species_name: Optional[str] = None,
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

        annotations = None
        species_list = []

        # try:
        df_species = basico.model_info.get_species(model=model_object.copasi_model)

        if species_name:

            if species_name not in df_species.index:
                error_message = f"The species '{species_name}' you provided was not found in the model data. This could be due to an incorrect species name (e.g., spelling or capitalization errors), the species not being included in the model. Please verify that the species name is correct and exists in the model. If the species is missing, you may need to update the model or check its documentation to ensure the species is included"
                return {"error": error_message}
                #raise _handle_error(f"Error: The Species '{species_name}'  which you have provided is not found in the model data.Please provide a valid species name which is present in the model.")
                #return {"message": f"Error: The Species '{species_name}' which you have provided is not found in the model data. Please provide a valid species name which is present in the model."}
                #st.write(f"Error: The Species '{species_name}' which you have provided is not found in the model data. Please provide a valid species name which is present in the model.")
                #return {"message": f"Error: The Species '{species_name}' which you have provided is not found in the model data. Please provide a valid species name which is present in the model."}
            # For single species, fetch the annotation
            annotation = basico.get_miriam_annotation(name=species_name)

            #if annotation is None:
            #   return {"message": f"Error: The Species '{species_name}' which you have provided is not found in the model data. Please provide a valid species name which is present in the model."}

               # raise ValueError(f"The Annotations for species '{species_name}' which you have provided is could not be found.Please provide a valid species name which is present in the model")
            #species_list.append(annotation)
            species_list = [annotation]
            print(type(species_list))
            annotations = annotation  # Set annotations for case where species_name is provided
            
            # Extract relevant columns (id, uri, resource, qualifier)
            descriptions = annotation['descriptions']
            data = []
            for desc in descriptions:
                data.append({
                    'species name': species_name,
                    'id': desc['id'],
                    'uri': desc['uri'],
                    'qualifier': desc['qualifier']
                })
            
            # Convert to DataFrame for consistent formatting
            annotations_df = pd.DataFrame(data)
            print(type(annotations_df))

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
            
            # st.session_state.messages.append({
            #     "type": "dataframe",
            #     "content": df1
            # })
            # st.dataframe(df1, use_container_width=True)
        

        #Case 2: If no species_name is provided, fetch all annotations for the model
        elif species_name is None:

            species= []
            df_species = basico.model_info.get_species(model=model_object.copasi_model)
            species = df_species.index.tolist()
            # if species_name not in species.index:
            #     error_message = f"The species '{species_name}' you provided was not found in the model data. This could be due to an incorrect species name (e.g., spelling or capitalization errors), the species not being included in the model. Please verify that the species name is correct and exists in the model. If the species is missing, you may need to update the model or check its documentation to ensure the species is included"
            #     return {"error": error_message}
            # species = ','.join(species)
            #print(species)

            # annotations = basico.get_miriam_annotation(name = species)  # Get all annotations for the model
            # species_list = annotations.get("descriptions", [])
            # data = []
            # for desc in species_list:
            #     data.append({
            #         'species name': species,
            #         'id': desc['id'],
            #         'uri': desc['uri'],
            #         'resource': desc['resource'],
            #         'qualifier': desc['qualifier']
            #     })
            all_annotations_data = []
# Iterate over each species and fetch its annotations
            for species_name in species:
                # Fetch annotations for the individual species
                annotations = basico.get_miriam_annotation(name=species_name)
                # print(annotations)
                    
                # Get the descriptions (annotations) for the species
                species_list = annotations.get("descriptions", [])
                
                # Extract and store relevant information for each description
                for desc in species_list:
                    all_annotations_data.append({
                        'species name': species_name,
                        'id': desc['id'],
                        'uri': desc['uri'],
                        'resource': desc.get('resource', 'N/A'),  # Default to 'N/A' if resource is missing
                        'qualifier': desc.get('qualifier', 'N/A')  # Default to 'N/A' if qualifier is missing
                    })
            
            # Convert to DataFrame for consistent formatting
            #print(all_annotations_data)
            # Convert the annotations to a DataFrame
            annotations_df = pd.DataFrame(all_annotations_data)
            
            #print(annotations_df)
            prompt_content = f'''
                            Convert the input into a table:

                            The table must contain the following columns:
                            - #
                            - Species Name
                            - ID (Clickable)
                            - URI (Clickable)
                            - Qualifier

                            Additional Guidelines:
                            # - The column # must contain the row number starting from 1.
                            - Embed the URL for each ID and URI in the table in the markdown format.
                            - Keep the ID and URI columns clickable as it is in df

                            Input:
                            {annotations_df}
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
    # def _run(self,
    #         species_name: Optional[str] = None,
    #         sys_bio_model: ModelData = None,  # Adding species_name as an optional argument
    #         run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
    #     """
    #     Run the tool to fetch species annotations.

    #     Args:
    #         species_name (Optional[str]): Specific species name to fetch annotations for.
    #         sys_bio_model (ModelData): The model data containing model ID, SBML file path, or model object.
    #         run_manager (Optional[CallbackManagerForToolRun]): The CallbackManagerForToolRun object.

    #     Returns:
    #         dict: A dictionary containing the species list.
    #     """
    #     st_session_key = self.st_session_key

    #     modelid = sys_bio_model.modelid if sys_bio_model is not None else None
    #     sbml_file_path = sys_bio_model.sbml_file_path if sys_bio_model is not None else None
    #     if st_session_key:
    #         if st_session_key not in st.session_state:
    #             return f"Session key {st_session_key} not found in Streamlit session state."
    #         if 'sbml_file_path' in st.session_state:
    #             sbml_file_path = st.session_state.sbml_file_path
    #     # Check if both modelid and sbml_file_path are None
    #     if modelid is None and sbml_file_path is None:
    #         if st_session_key:
    #             model_object = st.session_state[st_session_key]
    #             if model_object is None:
    #                 return "Please provide a BioModels ID or an SBML file path for simulation."
    #             modelid = model_object.model_id
    #         else:
    #             return "Please provide a BioModels ID or an SBML file path for simulation."
    #     elif modelid:
    #         model_object = BasicoModel(model_id=modelid)
    #         st.session_state[st_session_key] = model_object
    #     elif sbml_file_path:
    #         model_object = BasicoModel(sbml_file_path=sbml_file_path)
    #         modelid = model_object.model_id
    #         st.session_state[st_session_key] = model_object

    #     annotations = None
    #     species_list = []

    #     df_species = basico.model_info.get_species(model=model_object.copasi_model)
    #     all_annotations = []
    #     # Case when species_name is provided
    #     if species_name:
    #         if species_name not in df_species.index:
    #             error_message = f"The species '{species_name}' you provided was not found in the model data. Please verify that the species name is correct."
    #             return {"error": error_message}

    #         annotation = basico.get_miriam_annotation(name=species_name)
    #         all_annotations.extend(("descriptions", annotation))
    #         # species_list = [annotation]
    #         # annotations = annotation

    #         descriptions = annotation['descriptions']
    #         data = []
    #         for desc in descriptions:
    #             data.append({
    #                 'species name': species_name,
    #                 'id': desc['id'],
    #                 'uri': desc['uri'],
    #                 'qualifier': desc['qualifier']
    #             })

    #         annotations_df = pd.DataFrame(data)

    #     # Case when no species_name is provided, fetch for all species
    #     else:
    #         species = df_species.index.tolist()

    #         all_annotations_data = []
    #         for species_name in species:
    #             annotations = basico.get_miriam_annotation(name=species_name)
    #             species_list = annotations.get("descriptions", [])
    #             for desc in species_list:
    #                 all_annotations_data.append({
    #                     'species name': species_name,
    #                     'id': desc['id'],
    #                     'uri': desc['uri'],
    #                     'resource': desc.get('resource', 'N/A'),
    #                     'qualifier': desc.get('qualifier', 'N/A')
    #                 })

    #         annotations_df = pd.DataFrame(all_annotations_data)

    #     # Create the prompt content for formatting
    #     prompt_content = f'''
    #                     Convert the input data into a single table:

    #                     The table must contain the following columns:
    #                     - #
    #                     - Species Name
    #                     - ID (Clickable)
    #                     - URI (Clickable)
    #                     - Qualifier

    #                     Additional Guidelines:
    #                     - The column # must contain the row number starting from 1.
    #                     - Embed the URL for each ID and URI in the table in markdown format.
    #                     - Keep the ID and URI columns clickable.
    #                     - Combine all the tables into a single table.
    #                     - Put all the data in one table.

    #                     Input:
    #                     {annotations_df}
    #                     '''

    #     prompt_template = ChatPromptTemplate.from_messages(
    #         [("system", prompt_content),
    #         ("user", "{input}")]
    #     )

    #     llm = ChatOpenAI(model="gpt-4o-mini")
    #     parser = StrOutputParser()
    #     chain = prompt_template | llm | parser

    #     # Invoke the chain to format the annotations_df
    #     return chain.invoke({"input": annotations_df})
