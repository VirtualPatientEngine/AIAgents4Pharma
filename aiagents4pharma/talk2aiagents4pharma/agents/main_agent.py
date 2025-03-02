#/usr/bin/env python3

'''
This is the main agent file for the AIAgents4Pharma.
'''

import logging
import hydra
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from ...talk2biomodels.agents.t2b_agent import get_app as get_app_t2b
from ...talk2knowledgegraphs.agents.t2kg_agent import get_app as get_app_t2kg
from ..states.state_talk2aiagents4pharma import Talk2AIAgents4Pharma

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id,
            llm_model: BaseChatModel = ChatOpenAI(model='gpt-4o-mini', temperature=0)):
    '''
    This function returns the langraph app.
    '''
    # Load hydra configuration
    logger.log(logging.INFO, "Launching AIAgents4Pharma_Agent with thread_id %s", uniq_id)
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['agents/main_agent=default'])
        cfg = cfg.agents.main_agent
    logger.log(logging.INFO, "System_prompt of T2AA4P: %s", cfg.system_prompt)
    # Create supervisor workflow
    workflow = create_supervisor(
        [
            get_app_t2b(uniq_id, llm_model),    # Talk2BioModels
            get_app_t2kg(uniq_id, llm_model)    # Talk2KnowledgeGraphs
        ],
        model=llm_model,
        state_schema=Talk2AIAgents4Pharma,
        # Full history is needed to extract
        # the tool artifacts
        output_mode="full_history",
        add_handoff_back_messages=False,
        prompt=cfg.system_prompt
    )

    # Compile and run
    app = workflow.compile(checkpointer=MemorySaver(),
                           name="AIAgents4Pharma_Agent")

    return app
