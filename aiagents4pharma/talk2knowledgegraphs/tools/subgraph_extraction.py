"""
Tool for performing subgraph extraction.
"""

from typing import Type, Annotated
import logging
import pickle
import numpy as np
import pandas as pd
import hydra
import networkx as nx
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
import torch
from torch_geometric.data import Data
from ..utils.extractions.pcst import PCSTPruningMultiModal
from ..utils.embeddings.ollama import EmbeddingWithOllama
# from .load_arguments import ArgumentData

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubgraphExtractionInput(BaseModel):
    """
    SubgraphExtractionInput is a Pydantic model representing an input for extracting a subgraph.

    Args:
        prompt: Prompt to interact with the backend.
        tool_call_id: Tool call ID.
        state: Injected state.
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

                qa_chain = create_stuff_documents_chain(
                    state["llm_model"], prompt_template
                )
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

    def prepare_final_subgraph(
        self, subgraph: dict, pyg_graph: Data, textualized_graph: pd.DataFrame
    ) -> dict:
        """
        Prepare the subgraph based on the extracted subgraph.

        Args:
            subgraph: The extracted subgraph.
            pyg_graph: The PyTorch Geometric graph.
            textualized_graph: The textualized graph.

        Returns:
            A dictionary containing the PyG graph, NetworkX graph, and textualized graph.
        """
        # print(subgraph)
        # Prepare the PyTorch Geometric graph
        mapping = {n: i for i, n in enumerate(subgraph["nodes"].tolist())}
        pyg_graph = Data(
            # Node features
            # x=pyg_graph.x[subgraph["nodes"]],
            x=[pyg_graph.x[n] for n in subgraph["nodes"]], # Since diverse embedding dimensions
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

        return {
            "graph_pyg": pyg_graph,
            "graph_nx": nx_graph,
            "graph_text": textualized_graph,
        }

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
        extraction_name: str,
    ) -> Command:
        """
        Run the subgraph extraction tool.

        Args:
            tool_call_id: The tool call ID for the tool.
            state: Injected state for the tool.
            prompt: The prompt to interact with the backend.
            extraction_name: The name assigned to the subgraph extraction process.

        Returns:
            Command: The command to be executed.
        """
        logger.log(logging.INFO, "Invoking subgraph_extraction tool")

        # Load hydra configuration
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/subgraph_extraction=default"]
            )
            cfg = cfg.tools.subgraph_extraction

        # Retrieve source graph from the state
        graph_data = {}
        graph_data["source"] = state["dic_source_graph"][-1]  # The last source graph as of now
        # logger.log(logging.INFO, "Source graph: %s", source_graph)

        # Load the knowledge graph
        with open(graph_data["source"]["kg_pyg_path"], "rb") as f:
            graph_data["pyg"] = pickle.load(f)
        with open(graph_data["source"]["kg_text_path"], "rb") as f:
            graph_data["text"] = pickle.load(f)

        # Load the graph extraction data
        graph_ext = {dic["name"]: dic for dic in state["dic_extracted_graph"]}
        # logger.log(logging.INFO, "Extracted graph: %s", graph_ext)

        # Prepare prompt construction along with a list of endotypes
        if len(state["uploaded_files"]) != 0 and "endotype" in [
            f["file_type"] for f in state["uploaded_files"]
        ]:
            prompt = self.perform_endotype_filtering(prompt, state, cfg)

        # Prepare embedding model and embed the user prompt as query
        prompt_emb = torch.tensor(
            EmbeddingWithOllama(model_name=cfg.ollama_embeddings[0]).embed_query(prompt)
        ).float()

        # Prepare entity embeddings
        graph_data["ner_node_idx"] = [
            n["node_id"] for n in graph_ext[extraction_name]["ner_nodes"]
        ]
        # Construct a pandas dataframe over nodes
        graph_data["nodes_df"] = pd.DataFrame(
            {
                "node_id": graph_data["pyg"].node_id,
                "node_name": graph_data["pyg"].node_name,
                "node_type": graph_data["pyg"].node_type,
                "x": graph_data["pyg"].x,
            }
        )
        graph_data["query_df"] = graph_data["nodes_df"][
            graph_data["nodes_df"].node_id.isin(graph_data["ner_node_idx"])
        ].reset_index(drop=True)
        logger.log(logging.INFO, "Len of NER Node Index: %s", len(graph_data["ner_node_idx"]))
        logger.log(logging.INFO, "Shape of Query DF: %s", graph_data["query_df"].shape)

        # Prepare the PCSTPruning object and extract the subgraph
        # Parameters were set in the configuration file obtained from Hydra
        subgraph = PCSTPruningMultiModal(
            state["topk_nodes"],
            state["topk_edges"],
            cfg.cost_e,
            cfg.c_const,
            cfg.root,
            cfg.num_clusters,
            cfg.pruning,
            cfg.verbosity_level,
        ).extract_subgraph(graph_data["pyg"], graph_data["query_df"], prompt_emb)
        # logger.log(logging.INFO, "Subgraph extracted: %s", subgraph)
        logger.log(logging.INFO, "Subgraph nodes: %s", subgraph["nodes"])
        logger.log(logging.INFO, "Subgraph nodes length: %s", subgraph["nodes"].shape)
        logger.log(logging.INFO, "Subgraph edges: %s", subgraph["edges"])
        logger.log(logging.INFO, "Subgraph edges length: %s", subgraph["edges"].shape)

        # Prepare subgraph as a NetworkX graph and textualized graph
        final_subgraph = self.prepare_final_subgraph(
            subgraph, graph_data["pyg"], graph_data["text"]
        )

        # Store the response as graph_summary in the extracted graph
        for key, value in graph_ext.items():
            if key == extraction_name:
                value["topk_nodes"] = state["topk_nodes"]
                value["topk_edges"] = state["topk_edges"]
                value["graph_dict"] = {
                    "nodes": list(final_subgraph["graph_nx"].nodes(data=True)),
                    "edges": list(final_subgraph["graph_nx"].edges(data=True)),
                }
                value["graph_text"] = final_subgraph["graph_text"]

        # Prepare the dictionary of updated state
        dic_updated_state_for_model = {}
        for key, value in {
            "dic_extracted_graph": list(graph_ext.values()),
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        # Return the updated state of the tool
        return Command(
            update=dic_updated_state_for_model
            | {
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"Subgraph Extraction Result of {extraction_name}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
