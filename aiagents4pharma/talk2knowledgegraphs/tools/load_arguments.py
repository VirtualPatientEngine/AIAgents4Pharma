"""
A utility module for defining the dataclasses
for the arguments to set up initial settings
"""

from dataclasses import dataclass
from typing import Annotated


@dataclass
class ArgumentData:
    """
    Dataclass for storing the argument data.
    """

    extraction_name: Annotated[
        str,
        """An AI assigned _ separated name of the subgraph extraction
                                    based on human query and the context of the graph reasoning
                                    experiment.
                                    This must be set before the subgraph extraction is invoked.""",
    ]
