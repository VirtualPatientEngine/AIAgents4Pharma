#/usr/bin/env python3

'''
This is the agent file for the Talk2BioModels agent.
'''

import logging
from typing import Annotated
import hydra
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode, InjectedState
from ..tools.search_models import SearchModelsTool
from ..tools.get_modelinfo import GetModelInfoTool
from ..tools.simulate_model import SimulateModelTool
from ..tools.custom_plotter import CustomPlotterTool
from ..tools.ask_question import AskQuestionTool
from ..states.state_talk2biomodels import Talk2Biomodels

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model='gpt-4o-mini'):
    '''
    This function returns the langraph app.
    '''
    def agent_t2b_node(state: Annotated[dict, InjectedState]):
        '''
        This function calls the model.
        '''
        logger.log(logging.INFO, "Calling t2b_agent node with thread_id %s", uniq_id)
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return response

    # Define the tools
    simulate_model = SimulateModelTool()
    custom_plotter = CustomPlotterTool()
    ask_question = AskQuestionTool()
    search_model = SearchModelsTool()
    get_modelinfo = GetModelInfoTool()
    tools = ToolNode([
                    simulate_model,
                    ask_question,
                    custom_plotter,
                    search_model,
                    get_modelinfo
                    ])

    # Define the model
    llm = ChatOpenAI(model=llm_model, temperature=0)
    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2BioModels agent.")
    with hydra.initialize(version_base=None, config_path="../../../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['aiagents4pharma/talk2biomodels/agents/t2b_agent=default'])
        cfg = cfg.aiagents4pharma.talk2biomodels.agents.t2b_agent
    logger.log(logging.INFO, "state_modifier: %s", cfg.state_modifier)
    # Create the agent
    model = create_react_agent(
                llm,
                tools=tools,
                state_schema=Talk2Biomodels,
                state_modifier=cfg.state_modifier,
                checkpointer=MemorySaver()
            )

    # Define a new graph
    workflow = StateGraph(Talk2Biomodels)

    # Define the two nodes we will cycle between
    workflow.add_node("agent_t2b", agent_t2b_node)

    # Set the entrypoint as the first node
    # This means that this node is the first one called
    workflow.add_edge(START, "agent_t2b")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory
    # when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO,
               "Compiled the graph with thread_id %s and llm_model %s",
               uniq_id,
               llm_model)

    return app
