#/usr/bin/env python3

'''
This is the agent file for the Talk2Cells graph.
'''

import logging
import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from ..tools.scp_agent.search_studies import search_studies
from ..tools.scp_agent.display_studies import display_studies
from ..states.state_talk2cells import Talk2Cells

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id):
    '''
    This function returns the langraph app.
    '''
    def agent_scp_node(state: Talk2Cells):
        '''
        This function calls the model.
        '''
        logger.log(logging.INFO, "Creating SCP_Agent node with thread_id %s", uniq_id)
        # Get the messages from the state
        # messages = state['messages']
        # Call the model
        # inputs = {'messages': messages}
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        # The response is a list of messages and may contain `tool calls`
        # We return a list, because this will get added to the existing list
        # return {"messages": [response]}
        return response

    # Define the tools
    # tools = [search_studies, display_studies]
    tools = ToolNode([search_studies, display_studies])

    # Create the LLM
    # And bind the tools to it
    # model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    # Create an environment variable to store the LLM model
    # Check if the environment variable AIAGENTS4PHARMA_LLM_MODEL is set
    # If not, set it to 'gpt-4o-mini'
    llm_model = os.getenv('AIAGENTS4PHARMA_LLM_MODEL', 'gpt-4o-mini')
    # print (f'LLM model: {llm_model}')
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model=llm_model, temperature=0)
    model = create_react_agent(
                            llm,
                            tools=tools,
                            state_schema=Talk2Cells,
                            state_modifier=(
                                            "You are Talk2Cells agent."
                                            ),
                            checkpointer=MemorySaver()
                        )

    # Define a new graph
    workflow = StateGraph(Talk2Cells)

    # Define the two nodes we will cycle between
    workflow.add_node("agent_scp", agent_scp_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent_scp")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO, "Compiled the graph")

    return app
