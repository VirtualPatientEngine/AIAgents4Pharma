'''
Test cases for the display_studies.
'''

# from ..tools.display_studies import display_studies
from aiagents4pharma.talk2cells.tools.display_studies import display_studies

def test_tool_display_studies():
    '''
    Test the tool display_studies.
    '''
    response = display_studies.invoke(input={
                                'state': {'search_table': 1},
                                })
    # Check if the response is 1
    assert response == 1
