#!/usr/bin/env python3

"""
Tool for asking a question about the simulation results.
"""

import logging
from typing import Type, Annotated, Literal
import hydra
import basico
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools.base import BaseTool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AskQuestionInput(BaseModel):
    """
    Input schema for the AskQuestion tool.
    """
    question: str = Field(description="question about the simulation and steady state results")
    experiment_name: str = Field(description="""Name assigned to the simulation
                                            or steady state analysis when the tool 
                                            simulate_model or steady_state is invoked.""")
    question_context: Literal["simulation", "steady_state"] = Field(
        description="Context of the question")
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints.
# BaseTool is a Pydantic class and not having type hints
# can lead to unexpected behavior.
class AskQuestionTool(BaseTool):
    """
    Tool for asking a question about the simulation or steady state results.
    """
    name: str = "ask_question"
    description: str = """A tool to ask question about the
                        simulation or steady state results."""
    args_schema: Type[BaseModel] = AskQuestionInput
    return_direct: bool = False

    def _run(self,
             question: str,
             experiment_name: str,
             question_context: Literal["simulation", "steady_state"],
             state: Annotated[dict, InjectedState]) -> str:
        """
        Run the tool.

        Args:
            question (str): The question to ask about the simulation or steady state results.
            state (dict): The state of the graph.
            experiment_name (str): The name assigned to the simulation or steady state analysis.

        Returns:
            str: The answer to the question.
        """
        logger.log(logging.INFO,
                   "Calling ask_question tool %s, %s, %s",
                   question,
                   question_context,
                   experiment_name)
        # Load hydra configuration
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['tools/ask_question=default'])
            cfg = cfg.tools.ask_question
        # Get the context of the question
        # and based on the context, get the data
        # and prompt content to ask the question
        if question_context == "steady_state":
            dic_context = state["dic_steady_state_data"]
            prompt_content = cfg.steady_state_prompt
        else:
            dic_context = state["dic_simulated_data"]
            prompt_content = cfg.simulation_prompt
        # Extract the
        dic_data = {}
        for data in dic_context:
            for key in data:
                if key not in dic_data:
                    dic_data[key] = []
                dic_data[key] += [data[key]]
        # Create a pandas dataframe of the data
        df_data = pd.DataFrame.from_dict(dic_data)
        # Extract the data for the experiment
        # matching the experiment name
        df = pd.DataFrame(
            df_data[df_data['name'] == experiment_name]['data'].iloc[0]
        )
        logger.log(logging.INFO, "Shape of the dataframe: %s", df.shape)
        # # Extract the model units
        # model_units = basico.model_info.get_model_units()
        # Update the prompt content with the model units
        prompt_content += "Following are the model units:\n"
        prompt_content += f"{basico.model_info.get_model_units()}\n\n"
        # Create a pandas dataframe agent
        df_agent = create_pandas_dataframe_agent(
                        state['llm_model'],
                        allow_dangerous_code=True,
                        agent_type='tool-calling',
                        df=df,
                        max_iterations=5,
                        include_df_in_prompt=True,
                        number_of_head_rows=df.shape[0],
                        verbose=True,
                        prefix=prompt_content)
        # Invoke the agent with the question
        llm_result = df_agent.invoke(question, stream_mode=None)
        # print (llm_result)
        return llm_result["output"]
