import logging
from typing import Literal

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config.config import config
from state.shared_state import talk2comp
from tools.s2 import s2_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SemanticScholarAgent:
    """
    Agent for interacting with Semantic Scholar using LangGraph and LangChain.
    """

    def __init__(self):
        """
        Initializes the SemanticScholarAgent with necessary configurations.
        """
        try:
            logger.info("Initializing S2 Agent...")
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # Create the tools agent using config prompt
            self.tools_agent = create_react_agent(
                self.llm,
                tools=s2_tools,
                state_schema=talk2comp,
                state_modifier=config.S2_AGENT_PROMPT,
            )

            def execute_tools(state: talk2comp) -> Command[Literal["__end__"]]:
                """
                Execute tools and return results.

                Args:
                    state (talk2comp): The current state of the conversation.

                Returns:
                    Command[Literal["__end__"]]: The command to execute next.
                """
                logger.info("Executing tools")
                try:
                    result = self.tools_agent.invoke(state)
                    logger.info("Tool execution completed")
                    return Command(
                        goto=END,
                        update={
                            "messages": result["messages"],
                            "papers": result.get("papers", []),
                            "is_last_step": True,
                        },
                    )
                except (requests.RequestException, ToolException) as e:
                    logger.error("API or tool error: %s", str(e))
                    return Command(
                        goto=END,
                        update={
                            "messages": [AIMessage(content=f"Error: {str(e)}")],
                            "is_last_step": True,
                        },
                    )
                except ValueError as e:
                    logger.error("Value error: %s", str(e))
                    return Command(
                        goto=END,
                        update={
                            "messages": [
                                AIMessage(content=f"Input validation error: {str(e)}")
                            ],
                            "is_last_step": True,
                        },
                    )

            # Create graph
            workflow = StateGraph(talk2comp)
            workflow.add_node("tools", execute_tools)
            workflow.add_edge(START, "tools")

            self.graph = workflow.compile()
            logger.info("S2 Agent initialized successfully")

        except Exception as e:
            logger.error("Initialization error: %s", str(e))
            raise

    def invoke(self, state):
        """
        Invokes the SemanticScholarAgent with the given state.

        Args:
            state (talk2comp): The current state of the conversation.

        Returns:
            dict: The result of the invocation, including messages and papers.
        """
        try:
            logger.info("Invoking S2 agent")
            return self.graph.invoke(state)
        except (requests.RequestException, ToolException) as e:
            logger.error("Network or tool error in S2 agent: %s", str(e))
            return {
                "messages": [AIMessage(content=f"Error in processing: {str(e)}")],
                "papers": [],
            }
        except ValueError as e:
            logger.error("Value error in S2 agent: %s", str(e))
            return {
                "messages": [AIMessage(content=f"Invalid input: {str(e)}")],
                "papers": [],
            }
        except RuntimeError as e:
            logger.error("Runtime error in S2 agent: %s", str(e))
            return {
                "messages": [AIMessage(content=f"Internal error: {str(e)}")],
                "papers": [],
            }


# Create a global instance
s2_agent = SemanticScholarAgent()
