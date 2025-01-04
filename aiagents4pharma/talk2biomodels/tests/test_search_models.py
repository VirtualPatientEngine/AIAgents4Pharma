'''
Test cases for search_models.py
'''

from ..tools.search_models import SearchModelsTool

def test_tool_search_models():
    '''
    Test the tool search_models.
    '''
    search_models = SearchModelsTool()
    response = search_models.run({'query': 'Crohns Disease'})
    # Check if the response contains the BioModel ID
    # BIOMD0000000537
    assert 'BIOMD0000000537' in response

def test_get_metadata():
    '''
    Test the get_metadata method of the SearchModelsTool class.
    '''
    metadata = SearchModelsTool().get_metadata()
    assert metadata["name"] == "search_models"
    assert metadata["description"] == "Search models based on search query."
