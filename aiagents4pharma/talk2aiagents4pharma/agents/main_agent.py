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

def get_app(uniq_id, llm_model: BaseChatModel):
    '''
    This function returns the langraph app.
    '''
    if hasattr(llm_model, 'model_name'):
        if llm_model.model_name == 'gpt-4o-mini':
            llm_model = ChatOpenAI(model='gpt-4o-mini',
                                temperature=0,
                                model_kwargs={"parallel_tool_calls": False})
    # Load hydra configuration
    logger.log(logging.INFO, "Launching AIAgents4Pharma_Agent with thread_id %s", uniq_id)
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name='config',
                            overrides=['agents/main_agent=default'])
        cfg = cfg.agents.main_agent
    logger.log(logging.INFO, "System_prompt of T2AA4P: %s", cfg.system_prompt)
    with hydra.initialize(version_base=None, config_path="../../talk2biomodels/configs"):
        cfg_t2b = hydra.compose(config_name='config',
                            overrides=['agents/t2b_agent=default'])
        cfg_t2b = cfg_t2b.agents.t2b_agent
    with hydra.initialize(version_base=None, config_path="../../talk2knowledgegraphs/configs"):
        cfg_t2kg = hydra.compose(config_name='config',
                            overrides=['agents/t2kg_agent=default'])
        cfg_t2kg = cfg_t2kg.agents.t2kg_agent
    system_prompt = cfg.system_prompt
    system_prompt += "\n\nHere is the system prompt of T2B agent\n"
    system_prompt += cfg_t2b.state_modifier
    system_prompt += "\n\nHere is the system prompt of T2KG agent\n"
    system_prompt += cfg_t2kg.state_modifier
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
        add_handoff_back_messages=True,
        prompt=system_prompt
    )

    # Compile and run
    app = workflow.compile(checkpointer=MemorySaver(),
                           name="AIAgents4Pharma_Agent")

    return app
