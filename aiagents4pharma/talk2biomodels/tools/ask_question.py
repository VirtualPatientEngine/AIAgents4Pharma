#!/usr/bin/env python3

"""
Tool for asking a question about the simulation results.
"""

import logging
from typing import Type, Annotated
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools.base import BaseTool
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AskQuestionInput(BaseModel):
    """
    Input schema for the AskQuestion tool.
    """
    question: str = Field(description="question about the simulation results")
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints.
# BaseTool is a Pydantic class and not having type hints
# can lead to unexpected behavior.
class AskQuestionTool(BaseTool):
    """
    Tool for calculating the product of two numbers.
    """
    name: str = "ask_question"
    description: str = "A tool to ask question about the simulation results."
    args_schema: Type[BaseModel] = AskQuestionInput
    return_direct: bool = False

    def _run(self,
             question: str,
             state: Annotated[dict, InjectedState]) -> str:
        """
        Run the tool.

        Args:
            question (str): The question to ask about the simulation results.
            state (dict): The state of the graph.
            run_manager (Optional[CallbackManagerForToolRun]): The CallbackManagerForToolRun object.

        Returns:
            str: The answer to the question.
        """
        logger.log(logging.INFO,
                   "Calling ask_question tool %s", question)
        # Check if the simulation results are available
        if 'dic_simulated_data' not in state:
            return "Please run the simulation first before \
                asking a question about the simulation results."
        df = pd.DataFrame.from_dict(state['dic_simulated_data'])
        prompt_content = None
        # if run_manager and 'prompt' in run_manager.metadata:
        #     prompt_content = run_manager.metadata['prompt']
        # Create a pandas dataframe agent with OpenAI
        df_agent = create_pandas_dataframe_agent(
                        ChatOpenAI(model=state['llm_model']),
                        allow_dangerous_code=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        df=df,
                        prefix=prompt_content)
        llm_result = df_agent.invoke(question)
        return llm_result["output"]
