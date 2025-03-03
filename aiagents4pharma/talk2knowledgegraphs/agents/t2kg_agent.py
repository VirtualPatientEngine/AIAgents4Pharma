'''
This is the agent file for the Talk2KnowledgeGraphs agent.
'''

import logging
from typing import Annotated
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode, InjectedState
from ..tools.subgraph_extraction import SubgraphExtractionTool
from ..tools.subgraph_summarization import SubgraphSummarizationTool
from ..tools.graphrag_reasoning import GraphRAGReasoningTool
from ..states.state_talk2knowledgegraphs import Talk2KnowledgeGraphs

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model: BaseChatModel):
    '''
    This function returns the langraph app.
    '''
    def agent_t2kg_node(state: Annotated[dict, InjectedState]):
        '''
        This function calls the model.
        '''
        logger.log(logging.INFO, "Calling t2kg_agent node with thread_id %s", uniq_id)
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return response

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2KnowledgeGraphs agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['agents/t2kg_agent=default'])
        cfg = cfg.agents.t2kg_agent

    # Define the tools
    subgraph_extraction = SubgraphExtractionTool()
    subgraph_summarization = SubgraphSummarizationTool()
    graphrag_reasoning = GraphRAGReasoningTool()
    tools = ToolNode([
                    subgraph_extraction,
                    subgraph_summarization,
                    graphrag_reasoning,
                    ])

    # Create the agent
    model = create_react_agent(
                llm_model,
                tools=tools,
                state_schema=Talk2KnowledgeGraphs,
                prompt=cfg.state_modifier,
                version='v2',
                checkpointer=MemorySaver()
            )

    # Define a new graph
    workflow = StateGraph(Talk2KnowledgeGraphs)

    # Define the two nodes we will cycle between
    workflow.add_node("agent_t2kg", agent_t2kg_node)

    # Set the entrypoint as the first node
    # This means that this node is the first one called
    workflow.add_edge(START, "agent_t2kg")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory
    # when compiling the graph
    app = workflow.compile(checkpointer=checkpointer,
                           name="T2KG_Agent")
    logger.log(logging.INFO,
               "Compiled the graph with thread_id %s and llm_model %s",
               uniq_id,
               llm_model)

    return app
