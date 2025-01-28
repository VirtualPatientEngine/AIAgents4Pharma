"""
Tool for performing Graph RAG reasoning.
"""

import os
import logging
# import streamlit as st
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.prebuilt import InjectedState
import hydra

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGInput(BaseModel):
    """
    GraphRAGInput is a Pydantic model representing an input for Graph RAG reasoning.

    Args:
        prompt (str): Prompt to interact with the backend.
        subgraph_context (str): Subgraph context for the chat request.
    """
    prompt: str = Field(description="Prompt to interact with the backend.")
    subgraph_context: str = Field(description="Subgraph context for the chat request.")

class GraphRAGTool(BaseTool):
    """
    This tool performs reasoning using a Graph Retrieval-Augmented Generation (RAG) approach
    over user's request by considering textualized subgraph context and document context.
    """
    name: str = "graphrag_reasoning"
    description: str = "A tool to perform reasoning using a Graph RAG approach."
    args_schema: Type[BaseModel] = GraphRAGInput

    def setup_vector_store(self, vector_store, vector_store_path, uploaded_files, cfg):
        """
        Prepare documents from uploaded files and create vector store out of them.

        Args:
            vector_store (Chroma): The vector store object.
            vector_store_path (str): The vector store path.
            uploaded_files (list): The list of uploaded files.
            cfg:The Hydra configuration.

        Returns:
            vector_store: The vector store object.
        """
        # Prepare documents from uploaded files and create vector store out of them
        all_docs = []
        for uploaded_file in uploaded_files:
            if uploaded_file["file_type"] == "drug_data":
                # Perform a simple search wheter the uploaded doc is already stored in the DB
                # Split the docs if it is not existed yet
                logger.log(logging.INFO,
                           "Check if %s is already stored in the DB", uploaded_file['file_path'])
                if (len(vector_store.get(
                            where={"source": uploaded_file['file_path']}, include=["metadatas"]
                        )["ids"])== 0):
                    logger.log(logging.INFO,
                               "%s is not exists, store it into DB.", uploaded_file['file_path'])

                    # Load documents
                    raw_documents = PyPDFLoader(file_path=uploaded_file['file_path']).load()

                    # Split documents
                    # May need to find an optimal chunk size and overlap configuration
                    documents = CharacterTextSplitter(
                        separator=cfg.splitter_separator,
                        chunk_size=cfg.splitter_chunk_size,
                        chunk_overlap=cfg.splitter_chunk_overlap
                    ).split_documents(raw_documents)
                    # Add source file path to the metadata
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file['file_path']

                    # Add documents to the list
                    all_docs.extend(documents)

        if len(all_docs) > 0:
            # Create vector store from split documents
            vector_store = Chroma.from_documents(
                all_docs,
                embeddings,
                persist_directory=vector_store_path,
            )

        return vector_store

    def _run(self, state: Annotated[dict, InjectedState], prompt: str, subgraph_context: str):
        """
        Run the Graph RAG reasoning tool.

        Args:
            state: The injected state.
            prompt: The prompt to interact with the backend.
            subgraph_context: The subgraph context for the chat request.
        """
        # Load Hydra configuration
        logger.log(logging.INFO, "Load Hydra configuration for graphrag reasoning")
        with hydra.initialize(version_base=None, config_path="../../../../../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['web/backend/routers/tools/graphrag=default'])
            cfg = cfg.web.backend.routers.tools.graphrag

        # Prepare embeddings and LLM based on the model name
        logger.log(logging.INFO, "Prepare embeddings and LLM")
        model_name = state["llm_model"]
        if model_name in cfg.openai_llms:
            embeddings = OpenAIEmbeddings(model=cfg.openai_embeddings[0],
                                        api_key=cfg.openai_api_key)
            llm = ChatOpenAI(model=model_name,
                            api_key=cfg.openai_api_key,
                            temperature=cfg.temperature,
                            streaming=cfg.streaming)
        else:
            embeddings = OllamaEmbeddings(model=cfg.ollama_embeddings[0])
            llm = ChatOllama(model=model_name,
                            temperature=cfg.temperature,
                            streaming=cfg.streaming)

        # Load existing vector store from the directory
        logger.log(logging.INFO, "Load existing vector store")
        vector_store_path = os.path.join(
            cfg.vectordb_dir,
            f'{cfg.vectordb_prefix}_{model_name.replace(":", "_")}'
        )
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)

        if len(state["uploaded_files"]) != 0:
            # Populate docs
            logger.log(logging.INFO, "Initiate indexing over docs into vector store")
            vector_store = self.setup_vector_store(vector_store,
                                                   vector_store_path,
                                                   state["uploaded_files"],)

        # Set prompt template
        logger.log(logging.INFO, "Prepare contextualized que prompt")
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", cfg.prompt_graphrag_w_docs_context),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        # Set another prompt template
        logger.log(logging.INFO, "Prepare prompt template")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", cfg.prompt_graphrag_w_docs),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Prepare chain with retrieved documents
        logger.log(logging.INFO, "Chain setup")
        chain = create_retrieval_chain(
            create_history_aware_retriever(
                llm,
                vector_store.as_retriever(search_type=cfg.retriever_search_type,
                                        search_kwargs={'k': cfg.retriever_k,
                                                        'fetch_k': cfg.retriever_fetch_k,
                                                        'lambda_mult': cfg.retriever_lambda_mult}),
                                                        contextualize_q_prompt),
                                                        create_stuff_documents_chain(llm,
                                                                                    prompt_template))

        # Invoke the chain
        logger.log(logging.INFO, "Return the ouput after invoking the chain")
        response = chain.invoke({
            "input": prompt,
            "chat_history": [
                SystemMessage(content=m[1]) if m[0] == "system" else
                HumanMessage(content=m[1]) if m[0] == "human" else
                AIMessage(content=m[1])
                for m in state["history"]],
            "subgraph_context": subgraph_context,
        })

        return response["answer"]
