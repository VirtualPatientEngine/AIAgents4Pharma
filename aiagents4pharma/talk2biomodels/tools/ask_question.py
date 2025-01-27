#!/usr/bin/env python3

"""
Tool for asking a question about the simulation results.
"""

import logging
from typing import Type, Annotated, Literal
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
        # print (f'Calling ask_question tool {question}, {question_context}, {experiment_name}')
        if question_context == "steady_state":
            dic_context = state["dic_steady_state_data"]
        else:
            dic_context = state["dic_simulated_data"]
        dic_data = {}
        for data in dic_context:
            for key in data:
                if key not in dic_data:
                    dic_data[key] = []
                dic_data[key] += [data[key]]
        # print (dic_data)
        df_data = pd.DataFrame.from_dict(dic_data)
        df = pd.DataFrame(
            df_data[df_data['name'] == experiment_name]['data'].iloc[0]
        )
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
