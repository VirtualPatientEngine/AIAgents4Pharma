"""
Tool for performing Graph RAG reasoning.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
import hydra

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAGReasoningInput(BaseModel):
    """
    GraphRAGReasoningInput is a Pydantic model representing an input for Graph RAG reasoning.

    Args:
        state: Injected state.
        prompt: Prompt to interact with the backend.
        extraction_name: Name assigned to the subgraph extraction process
    """

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        description="Tool call ID."
    )
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")
    prompt: str = Field(description="Prompt to interact with the backend.")
    extraction_name: str = Field(
        description="""Name assigned to the subgraph extraction process
                    when the subgraph_extraction tool is invoked."""
    )


class GraphRAGReasoningTool(BaseTool):
    """
    This tool performs reasoning using a Graph Retrieval-Augmented Generation (RAG) approach
    over user's request by considering textualized subgraph context and document context.
    """

    name: str = "graphrag_reasoning"
    description: str = """A tool to perform reasoning using a Graph RAG approach
                        by considering textualized subgraph context and document context."""
    args_schema: Type[BaseModel] = GraphRAGReasoningInput

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
        extraction_name: str,
    ):
        """
        Run the Graph RAG reasoning tool.

        Args:
            tool_call_id: The tool call ID.
            state: The injected state.
            prompt: The prompt to interact with the backend.
            extraction_name: The name assigned to the subgraph extraction process.
        """
        logger.log(
            logging.INFO, "Invoking graphrag_reasoning tool for %s", extraction_name
        )

        # Load Hydra configuration
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/graphrag_reasoning=default"]
            )
            cfg = cfg.tools.graphrag_reasoning

        # Prepare documents
        all_docs = []
        if len(state["uploaded_files"]) != 0:
            for uploaded_file in state["uploaded_files"]:
                if uploaded_file["file_type"] == "drug_data":
                    # Load documents
                    raw_documents = PyPDFLoader(
                        file_path=uploaded_file["file_path"]
                    ).load()

                    # Split documents
                    # May need to find an optimal chunk size and overlap configuration
                    documents = RecursiveCharacterTextSplitter(
                        chunk_size=cfg.splitter_chunk_size,
                        chunk_overlap=cfg.splitter_chunk_overlap,
                    ).split_documents(raw_documents)

                    # Add documents to the list
                    all_docs.extend(documents)

        # Load the extracted graph
        extracted_graph = {dic["name"]: dic for dic in state["dic_extracted_graph"]}
        # logger.log(logging.INFO, "Extracted graph: %s", extracted_graph)

        # Set another prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", cfg.prompt_graphrag_w_docs), ("human", "{input}")]
        )

        # Prepare chain with retrieved documents
        qa_chain = create_stuff_documents_chain(state["llm_model"], prompt_template)
        rag_chain = create_retrieval_chain(
            InMemoryVectorStore.from_documents(
                documents=all_docs, embedding=state["embedding_model"]
            ).as_retriever(
                search_type=cfg.retriever_search_type,
                search_kwargs={
                    "k": cfg.retriever_k,
                    "fetch_k": cfg.retriever_fetch_k,
                    "lambda_mult": cfg.retriever_lambda_mult,
                },
            ),
            qa_chain,
        )

        # Invoke the chain
        response = rag_chain.invoke(
            {
                "input": prompt,
                "subgraph_summary": extracted_graph[extraction_name]["graph_summary"],
            }
        )

        return Command(
            update={
                # update the message history
                "messages": [ToolMessage(content=response, tool_call_id=tool_call_id)]
            }
        )
