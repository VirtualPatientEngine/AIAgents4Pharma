'''
Test cases for the search_studies
'''

# from ..tools.search_studies import search_studies
from aiagents4pharma.talk2cells.tools.search_studies import search_studies

def test_tool_search_studies():
    '''
    Test the tool search_studies.
    '''
    response = search_studies.invoke(input={
                                'search_term': 'Crohns Disease',
                                'tool_call_id': '12345',
                                })
    # Check if the key 'search_table' is a string
    assert isinstance(response.update['search_table'], str)
