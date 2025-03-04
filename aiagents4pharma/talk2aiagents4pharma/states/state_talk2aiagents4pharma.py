"""
This is the state file for the Talk2AIAgents4Pharma agent.
"""

from ...talk2biomodels.states.state_talk2biomodels import Talk2Biomodels
from ...talk2knowledgegraphs.states.state_talk2knowledgegraphs import Talk2KnowledgeGraphs

class Talk2AIAgents4Pharma(Talk2Biomodels,
                           Talk2KnowledgeGraphs):
    """
    The state for the Talk2AIAgents4Pharma agent.

    This class inherits from the classes:
    1. Talk2Biomodels
    2. Talk2KnowledgeGraphs
    """
