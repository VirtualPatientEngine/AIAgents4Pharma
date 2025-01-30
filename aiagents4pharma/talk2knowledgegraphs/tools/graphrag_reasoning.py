"""
Tool for performing Graph RAG reasoning.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.prebuilt import InjectedState
import hydra

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGReasoningInput(BaseModel):
    """
    GraphRAGReasoningInput is a Pydantic model representing an input for Graph RAG reasoning.

    Args:
        prompt: Prompt to interact with the backend.
        state: Injected state.
    """
    prompt: str = Field(description="Prompt to interact with the backend.")
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")

class GraphRAGReasoningTool(BaseTool):
    """
    This tool performs reasoning using a Graph Retrieval-Augmented Generation (RAG) approach
    over user's request by considering textualized subgraph context and document context.
    """
    name: str = "graphrag_reasoning"
    description: str = "A tool to perform reasoning using a Graph RAG approach."
    args_schema: Type[BaseModel] = GraphRAGReasoningInput

    def _run(self, state: Annotated[dict, InjectedState], prompt: str):
        """
        Run the Graph RAG reasoning tool.

        Args:
            state: The injected state.
            prompt: The prompt to interact with the backend.
        """
        # Load Hydra configuration
        logger.log(logging.INFO, "Loading Hydra configuration for graphrag reasoning")
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name='config',
                                overrides=['tools/graphrag_reasoning=default'])
            cfg = cfg.tools.graphrag_reasoning

        # Prepare embeddings and LLM based on the model name
        logger.log(logging.INFO, "Preparing embeddings and LLM")
        if state["llm_model"] in cfg.openai_llms:
            emb_model = OpenAIEmbeddings(model=cfg.openai_embeddings[0],
                                        api_key=cfg.openai_api_key)
            llm = ChatOpenAI(model=state["llm_model"], temperature=cfg.temperature)
        else:
            emb_model = OllamaEmbeddings(model=cfg.ollama_embeddings[0])
            llm = ChatOllama(model=state["llm_model"], temperature=cfg.temperature)

        # Load existing vector store from the directory
        logger.log(logging.INFO, "Loading documents")
        # Prepare documents from uploaded files and create vector store out of them
        if len(state["uploaded_files"]) != 0:
            all_docs = []
            for uploaded_file in state["uploaded_files"]:
                if uploaded_file["file_type"] == "drug_data":
                    # Load documents
                    raw_documents = PyPDFLoader(file_path=uploaded_file['file_path']).load()

                    # Split documents
                    # May need to find an optimal chunk size and overlap configuration
                    documents = RecursiveCharacterTextSplitter(
                        chunk_size=cfg.splitter_chunk_size,
                        chunk_overlap=cfg.splitter_chunk_overlap
                    ).split_documents(raw_documents)

                    # Add documents to the list
                    all_docs.extend(documents)

        # Set another prompt template
        logger.log(logging.INFO, "Preparing prompt template")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", cfg.prompt_graphrag_w_docs),
                ("human", "{input}")
            ]
        )

        # Prepare chain with retrieved documents
        logger.log(logging.INFO, "Chain setup")
        qa_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(
                    InMemoryVectorStore.from_documents(
                        documents=all_docs,
                        embedding=emb_model
                        ).as_retriever(search_type=cfg.retriever_search_type,
                                       search_kwargs={'k': cfg.retriever_k,
                                                      'fetch_k': cfg.retriever_fetch_k,
                                                      'lambda_mult': cfg.retriever_lambda_mult}),
                                        qa_chain)

        # Invoke the chain
        logger.log(logging.INFO, "Invoking chain")
        response = rag_chain.invoke({
            "input": prompt,
            "subgraph_summary": state["graph_summary"],
        })

        return response
