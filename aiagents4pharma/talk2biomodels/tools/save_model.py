#!/usr/bin/env python3

"""
Tool for saving models.
"""

import logging
import os
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SaveModelInput(BaseModel):
    """
    Input schema for the save model tool.
    """

    path_to_folder: str = Field(
        description="Path to folder to save the model. Keep it to . if not provided.", default="."
    )
    output_filename: str = Field(
        description="Filename to save the model as. Default is 'saved_model.xml'.",
        default="saved_model.xml",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SaveModelTool(BaseTool):
    """
    Tool for saving a model.
    """

    name: str = "save_model"
    description: str = "A tool to save the current biomodel to a \
                        user specified path with the default filename\
                         'saved_model.xml'"
    args_schema: type[BaseModel] = SaveModelInput
    return_direct: bool = False

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        path_to_folder: str = ".",
        output_filename: str = "saved_model.xml",
    ) -> Command:
        """
        Run the tool.

        Args:
            path (str): The path to save the model.
            tool_call_id (str): The tool call ID.

        Returns:

        """
        logger.log(
            logging.INFO,
            "Saving model to path: %s with filename: %s",
            path_to_folder,
            output_filename,
        )
        # Check if path does not exist
        if not os.path.exists(path_to_folder):
            content = f"Error: Path {path_to_folder} does not exist."
            logger.error(content)
        else:
            logger.info("Saving now")
            # Save the model to the specified path
            with open(os.path.join(path_to_folder, output_filename), "w", encoding="utf-8") as f:
                f.write(state["model_as_string"][-1])
            content = f"Model saved successfully to {path_to_folder}/{output_filename}."
            logger.info(content)
        # Return the updated state of the tool
        return Command(
            update={
                # update the message history
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
