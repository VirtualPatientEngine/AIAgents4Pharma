#!/usr/bin/env python3

'''
A tool to fetch studies from the Single Cell Portal.
'''

import logging
from typing import Annotated
import requests
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
import pandas as pd

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool('search_studies')
def search_studies(search_term: str,
                   tool_call_id: Annotated[str, InjectedToolCallId],
                   limit: int = 5):
    """
    Fetch studies from single cell portal

    Args:
        search_term (str): The search term to use. Example: "COVID-19", "cancer", etc.
        limit (int): The number of papers to return. Default is 5.

    """
    logger.log(logging.INFO, "Calling the tool search_studies")
    scp_endpoint = 'https://singlecell.broadinstitute.org/single_cell/api/v1/search?type=study'
    # params = {'terms': search_term, 'facets': 'MONDO_0005011'}
    params = {'terms': search_term}
    status_code = 0
    while status_code != 200:
        # Make a GET request to the single cell portal
        search_response = requests.get(scp_endpoint,
                                       params=params,
                                       timeout=10,
                                       verify=False)
        status_code = search_response.status_code
        logger.log(logging.INFO, "Status code %s received from SCP")

    # Select the columns to display in the table
    selected_columns = ["study_source", "name", "study_url", "gene_count", "cell_count"]

    # Extract the data from the response
    # with the selected columns
    df = pd.DataFrame(search_response.json()['studies'])[selected_columns]

    # Convert column 'Study Name' into clickable
    # hyperlinks from the column 'Study URL'
    scp_api_url = 'https://singlecell.broadinstitute.org'
    df['name'] = df.apply(
            lambda x: f"<a href=\"{scp_api_url}/{x['study_url']}\">{x['name']}</a>",
            axis=1)

    # Excldue the column 'Study URL' from the dataframe
    df = df.drop(columns=['study_url'])

    # Add a new column a the beginning of the dataframe with row numbers
    df.insert(0, 'S/N', range(1, 1 + len(df)))

    # Update the state key 'search_table' with the dataframe in markdown format
    return Command(
        update={
            # update the state keys
            "search_table": df.to_markdown(tablefmt="grid"),
            # update the message history
            "messages": [
                ToolMessage(
                    f"Successfully fetched {limit} studies on {search_term}.",
                    tool_call_id=tool_call_id
                )
            ],
        }
    )
