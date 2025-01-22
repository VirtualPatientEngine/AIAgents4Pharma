#/usr/bin/env python3

'''
Agent for interacting with Semantic Scholar
'''

import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from ..config.config import config
from ..state.state_talk2competitors import Talk2Competitors
# from ..tools.s2 import s2_tools
from ..tools.s2.search import search_tool
from ..tools.s2.display_results import display_results
from ..tools.s2.single_paper_rec import get_single_paper_recommendations
from ..tools.s2.multi_paper_rec import get_multi_paper_recommendations

load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model='gpt-4o-mini'):
    '''
    This function returns the langraph app.
    '''
    def agent_s2_node(state: Talk2Competitors):
        '''
        This function calls the model.
        '''
        logger.log(logging.INFO, "Creating Agent_S2 node with thread_id %s", uniq_id)
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return response

    # Define the tools
    tools = [search_tool,
            display_results,
            get_single_paper_recommendations,
            get_multi_paper_recommendations]

    # Create the LLM
    llm = ChatOpenAI(model=llm_model, temperature=0)
    model = create_react_agent(
                            llm,
                            tools=tools,
                            state_schema=Talk2Competitors,
                            state_modifier=config.S2_AGENT_PROMPT,
                            checkpointer=MemorySaver()
                        )

    # Define a new graph
    workflow = StateGraph(Talk2Competitors)

    # Define the two nodes we will cycle between
    workflow.add_node("agent_s2", agent_s2_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent_s2")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO, "Compiled the graph")

    return app
