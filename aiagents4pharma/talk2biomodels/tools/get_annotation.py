from typing import Type, List, Annotated,Tuple, Union, Literal, TypedDict,Optional
from dataclasses import dataclass
import logging
import pandas as pd
from pydantic import BaseModel, Field
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import BaseTool
from aiagents4pharma.talk2biomodels.api.uniprot import search_uniprot_labels
from aiagents4pharma.talk2biomodels.api.ols import search_ols_labels
from aiagents4pharma.talk2biomodels.api.kegg import fetch_kegg_annotations
import basico
from ..models.basico_model import BasicoModel
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from .load_biomodel import ModelData,load_biomodel
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import json

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GetAnnotationInput(BaseModel):
    """
    Input schema for annotation tool.
    """
    sys_bio_model: ModelData = Field(description="model data")
    tool_call_id: Annotated[str, InjectedToolCallId]
    # query: str = Field(description="User query about the annotations.")
    species_names: Optional[List[str]] = Field(default=None,
                                               description="List of species names to fetch annotations for.")
    state: Annotated[dict, InjectedState]

class GetAnnotationTool(BaseTool):
    """
    Tool for fetching species annotations based on the provided annotation type.
    """
    name: str = "get_annotation"
    description: str = "Fetches species annotations from the model."
    args_schema: Type[BaseModel] = GetAnnotationInput
    return_direct: bool = True

    def _run(self, 
            #  query: str,
             species_names: Optional[List[str]]=None,
             sys_bio_model: ModelData =None,
             tool_call_id: Annotated[str, InjectedToolCallId]=None,
             state: Annotated[dict, InjectedState] = None
             ) -> str:
        """
        Run the tool to fetch species annotations.
        """
        sbml_file_path = state['sbml_file_path'][-1] if len(state['sbml_file_path']) > 0 else None
        model_object = load_biomodel(sys_bio_model,
                                  sbml_file_path=sbml_file_path)

        # Fetch species information
        df_species = basico.model_info.get_species(model=model_object.copasi_model)
        
        species_not_found = []
        data = []       
        # Default to all species if none provided
        if species_names is None:
            species_names = df_species.index.tolist()

        # logger.info("Running GetAnnotationTool with species names: %s", species_names)


        for species in species_names:
            annotation = basico.get_miriam_annotation(name=species)
            if annotation is None:
                species_not_found.append(species)
                continue

            descriptions = annotation.get("descriptions", [])
            for desc in descriptions:
                data.append({
                    'Species Name': species,
                    'Link': desc['id'],
                    'Qualifier': desc['qualifier']
                })

        annotations_df = pd.DataFrame(data)

        if annotations_df.empty:
            species_msg = (
            f"None of the species entered were found. The following species do not exist: "
            f"{', '.join(species_not_found)}." if species_not_found else
            "No annotations found for the species entered."
            )
            return species_msg

        # if annotations_df.empty:
        #     species_msg = (
        #         f"None of the species entered were found. The following species do not exist: "
        #         f"{', '.join(species_names)}." if species_names else
        #         "No annotations found for the species entered."
        #     )
        #     return Command(
        #         update={
        #             "messages": [
        #                 ToolMessage(
        #                     content=species_msg,
        #                     tool_call_id=tool_call_id
        #                 )
        #             ]
        #         }
        #     )

        # if annotations_df.empty:
        #     species_msg = (
        #         f"None of the species entered were found. The following species do not exist: "
        #         f"{', '.join(species_not_found)}." if species_not_found else
        #         "No annotations found for the species entered."
        #     )
        #     return species_msg

        # Process annotations and fetch additional labels
        annotations_df['Id'] = annotations_df['Link'].str.split('/').str[-1]
        annotations_df['Database'] = annotations_df['Link'].str.split('/').str[-2]

        identifiers = annotations_df[['Id', 'Database']].to_dict(orient='records')
        descriptions = self._fetch_descriptions(identifiers)
        print(descriptions)

        annotations_df['Description'] = annotations_df['Id'].apply(lambda x: descriptions.get(x, '-'))
        annotations_df.index = annotations_df.index + 1

        
        # annotations_df['Id'] = annotations_df.apply(
        #     lambda row: f'<a href="{row["Link"]}" target="_blank">{row["Id"]}</a>',
        #     axis=1
        # )



        new_annotations_df = ["Species Name", "Description","Database","Id","Link", "Qualifier"]
        df = annotations_df[new_annotations_df]

        print(df)
        def process_link(link):
            substrings = ["chebi/", "pato/","pr/","fma/","sbo/"]  # Substrings to remove
            for substring in substrings:
                if substring in link:
                    link = link.replace(substring, "")  # Remove the substring
            if "kegg.compound" in link:
                link = link.replace("kegg.compound/", "kegg.compound:")  # Ensure `/` before and `:` after
            return link

        df["Link"] = df["Link"].apply(process_link)

        # Display the updated DataFrame
        print(df)


        # name_list = annotations_df["Species Name"].tolist()

        # class CustomHeader(TypedDict):
        #     """
        #     A list of species based on user query.
        #     """
        #     relevant_species: Union[None, List[Literal[*name_list]]] = Field(
        #         description="List of species based on user query. If no relevant species are found, it will be None."
        #     )

        # # llm_with_structured_output = llm.with_structured_output(CustomHeader) 
        # # results = llm_with_structured_output.invoke(query)   

        # # Initialize the OpenAI LLM
        # llm = ChatOpenAI(model=state['llm_model'], temperature=0)

        # # Construct the input prompt for the LLM
        # column_data = annotations_df.to_dict(orient='records')
        # prompt = (
        #     "You are given a table of annotations with the following data: "
        #     f"{column_data}. "
        #     "The user has asked the following query: '" + query + "'. "
        #     "Identify the rows from the table that match the user's query based on any relevant information in the table. "
        #     "Return the relevant rows as output. If no rows match, return an empty result."
        # )

        # # Call the LLM to interpret the query and return matching rows
        # try:
        #     response = llm.invoke(prompt)
        #     if isinstance(response, str):
        #         response_data = json.loads(response)
        #     else:
        #         response_data = response
        
        # # Validate that response_data is a list of rows
        #     if not isinstance(response_data, list):
        #         raise ValueError("Expected a list of matching rows, got invalid format.")
        # logger.info("Current state: %s", state.values)

        dic_updated_state_for_model = {}
        for key, value in {
                        "model_id": [sys_bio_model.biomodel_id],
                        "sbml_file_path": [sbml_file_path],
                        "dic_annotations_data": df.to_dict() if not df.empty else {}
                        }.items():
            if value:
                dic_updated_state_for_model[key] = value
    
        return Command(
        update=dic_updated_state_for_model | {
                # update the message history
                "dic_annotations_data": df.to_dict()if not df.empty else {},
                "messages": [
                    ToolMessage(
                        content = "All the requested annotations extracted successfully." 
                        if not species_not_found 
                        else
                                f"The following species do not exist, and hence their annotations were not extracted: "
                                f"{', '.join(species_not_found)}."
                                f"Remaining species annotations is in following table",
                        tool_call_id=tool_call_id
                        
                        )
                    ],
                }
        )



    
        # return (
        #     "All the requested annotations extracted successfully." if not species_not_found else
        #     f"The following species do not exist, and hence their annotations were not extracted: "
        #     f"{', '.join(species_not_found)}. Remaining species annotations are in the table."
        # )

    def _fetch_descriptions(self, data: List[dict[str, str]]) -> dict[str, str]:
        """
        Fetch protein names or labels based on the database type.

        Args:
            data (List[Dict[str, str]]): A list of dictionaries containing 'Id' and 'Database'.

        Returns:
            Dict[str, str]: A mapping of identifiers to their names or labels.
        """
        results = {}
        grouped_data = {}

        for entry in data:
            identifier = entry.get('Id')
            database = entry.get('Database', '').lower()
            if identifier and database:
                grouped_data.setdefault(database, []).append(identifier)
            else:
                results[identifier or "unknown"] = "-"

        for database, identifiers in grouped_data.items():
            try:
                if database == 'uniprot':
                    results.update(search_uniprot_labels(identifiers))
                elif database in {'pato', 'chebi', 'sbo', 'fma', 'pr'}:
                    annotations = search_ols_labels([{"Id": id_, "Database": database} for id_ in identifiers])
                    for id_ in identifiers:
                        results[id_] = annotations.get(database, {}).get(id_, "-")
                else:
                    data = [{"Id": identifier, "Database": "kegg.compound"} for identifier in identifiers]
                    annotations = fetch_kegg_annotations(data)
                    for identifier in identifiers:
                        results[identifier] = annotations.get(database, {}).get(identifier, "-")
            except Exception as e:
                logger.error(f"Error fetching data for database '{database}': {e}")
                for id_ in identifiers:
                    results[id_] = "-"

        return results
