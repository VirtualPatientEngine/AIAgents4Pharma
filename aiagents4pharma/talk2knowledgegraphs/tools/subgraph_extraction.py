"""
Tool for performing subgraph extraction.
"""

import logging
import pickle
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
import numpy as np
import hydra
import networkx as nx
import torch
from torch_geometric.data import Data
from ..utils.extractions.pcst import PCSTPruning
from ..utils.embeddings.ollama import EmbeddingWithOllama

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubgraphExtractionInput(BaseModel):
    """
    SubgraphExtractionInput is a Pydantic model representing an input for extracting a subgraph.

    Args:
        prompt (str): Prompt to interact with the backend.
        topk_nodes (int): Number of top nodes for subgraph extraction.
        topk_edges (int): Number of top edges for subgraph extraction.
    """

    prompt: str = Field(description="Prompt to interact with the backend.")
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        description="Tool call ID."
    )
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")


class SubgraphExtractionTool(BaseTool):
    """
    This tool performs subgraph extraction based on user's prompt by taking into account
    the top-k nodes and edges.
    """

    name: str = "subgraph_extraction"
    description: str = "A tool for subgraph extraction based on user's prompt."
    args_schema: Type[BaseModel] = SubgraphExtractionInput

    def perform_endotype_filtering(
        self,
        prompt: str,
        state: Annotated[dict, InjectedState],
        cfg: hydra.core.config_store.ConfigStore,
    ) -> str:
        """
        Perform endotype filtering based on the uploaded files and prepare the prompt.

        Args:
            prompt: The prompt to interact with the backend.
            state: Injected state for the tool.
            cfg: Hydra configuration object.
        """
        # Loop through the uploaded files
        all_genes = []
        for uploaded_file in state["uploaded_files"]:
            if uploaded_file["file_type"] == "endotype":
                # Load the PDF file
                docs = PyPDFLoader(file_path=uploaded_file["file_path"]).load()

                # Split the text into chunks
                splits = RecursiveCharacterTextSplitter(
                    chunk_size=cfg.splitter_chunk_size,
                    chunk_overlap=cfg.splitter_chunk_overlap,
                ).split_documents(docs)

                # Create a chat prompt template
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", cfg.prompt_endotype_filtering),
                        ("human", "{input}"),
                    ]
                )

                qa_chain = create_stuff_documents_chain(state["llm_model"], prompt_template)
                rag_chain = create_retrieval_chain(
                    InMemoryVectorStore.from_documents(
                        documents=splits, embedding=state["embedding_model"]
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
                results = rag_chain.invoke({"input": prompt})
                all_genes.append(results["answer"])

        # Prepare the prompt
        if len(all_genes) > 0:
            prompt = " ".join(
                [prompt, cfg.prompt_endotype_addition, ", ".join(all_genes)]
            )

        return prompt

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
    ) -> Command:
        """
        Run the subgraph extraction tool.

        Args:
            tool_call_id: The tool call ID for the tool.
            state: Injected state for the tool.
            prompt: The prompt to interact with the backend.

        Returns:
            Command: The command to be executed.
        """
        # Load hydra configuration
        logger.log(logging.INFO, "Loading Hydra configuration for subgraph extraction")
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/subgraph_extraction=default"]
            )
            cfg = cfg.tools.subgraph_extraction

        # Load the knowledge graph
        # with open(cfg.input_tkg, 'rb') as f:
        with open(state["input_tkg"], "rb") as f:
            pyg_graph = pickle.load(f)
        # with open(cfg.input_text_tkg, 'rb') as f:
        with open(state["input_text_tkg"], "rb") as f:
            textualized_graph = pickle.load(f)

        # Prepare prompt construction along with a list of endotypes
        if len(state["uploaded_files"]) != 0 and "endotype" in [
            f["file_type"] for f in state["uploaded_files"]
        ]:
            prompt = self.perform_endotype_filtering(
                prompt, state, cfg
            )

        # Prepare embedding model and embed the user prompt as query
        query_emb = torch.tensor(
            EmbeddingWithOllama(model_name=cfg.ollama_embeddings[0]).embed_query(prompt)
        ).float()

        # Prepare the PCSTPruning object and extract the subgraph
        # Parameters were set in the configuration file obtained from Hydra
        subgraph = PCSTPruning(
            state["topk_nodes"],
            state["topk_edges"],
            cfg.cost_e,
            cfg.c_const,
            cfg.root,
            cfg.num_clusters,
            cfg.pruning,
            cfg.verbosity_level,
        ).extract_subgraph(pyg_graph, query_emb)

        # Prepare the PyTorch Geometric graph
        mapping = {n: i for i, n in enumerate(subgraph["nodes"].tolist())}
        pyg_graph = Data(
            # Node features
            x=pyg_graph.x[subgraph["nodes"]],
            node_id=np.array(pyg_graph.node_id)[subgraph["nodes"]].tolist(),
            node_name=np.array(pyg_graph.node_id)[subgraph["nodes"]].tolist(),
            enriched_node=np.array(pyg_graph.enriched_node)[subgraph["nodes"]].tolist(),
            num_nodes=len(subgraph["nodes"]),
            # Edge features
            edge_index=torch.LongTensor(
                [
                    [
                        mapping[i]
                        for i in pyg_graph.edge_index[:, subgraph["edges"]][0].tolist()
                    ],
                    [
                        mapping[i]
                        for i in pyg_graph.edge_index[:, subgraph["edges"]][1].tolist()
                    ],
                ]
            ),
            edge_attr=pyg_graph.edge_attr[subgraph["edges"]],
            edge_type=np.array(pyg_graph.edge_type)[subgraph["edges"]].tolist(),
            relation=np.array(pyg_graph.edge_type)[subgraph["edges"]].tolist(),
            label=np.array(pyg_graph.edge_type)[subgraph["edges"]].tolist(),
            enriched_edge=np.array(pyg_graph.enriched_edge)[subgraph["edges"]].tolist(),
        )

        # Networkx DiGraph construction to be visualized in the frontend
        nx_graph = nx.DiGraph()
        for n in pyg_graph.node_name:
            nx_graph.add_node(n)
        for i, e in enumerate(
            [
                [pyg_graph.node_name[i], pyg_graph.node_name[j]]
                for (i, j) in pyg_graph.edge_index.transpose(1, 0)
            ]
        ):
            nx_graph.add_edge(
                e[0],
                e[1],
                relation=pyg_graph.edge_type[i],
                label=pyg_graph.edge_type[i],
            )

        # Prepare the textualized subgraph
        textualized_graph = (
            textualized_graph["nodes"].iloc[subgraph["nodes"]].to_csv(index=False)
            + "\n"
            + textualized_graph["edges"].iloc[subgraph["edges"]].to_csv(index=False)
        )

        # Store the result state dictionary
        return Command(
            update={
                "graph_dict": {
                    "nodes": list(nx_graph.nodes(data=True)),  # Include node attributes
                    "edges": list(nx_graph.edges(data=True)),  # Include edge attributes
                },
                "graph_text": textualized_graph,
                "messages": [
                    ToolMessage(
                        content=f"Subgraph Extraction Result: {textualized_graph}",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )
