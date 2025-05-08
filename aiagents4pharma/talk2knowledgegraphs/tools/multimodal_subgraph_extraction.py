"""
Tool for performing multimodal subgraph extraction.
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
from ..utils.extractions.multimodal_pcst import MultimodalPCSTPruning
from ..utils.embeddings.ollama import EmbeddingWithOllama
# from ..utils.embeddings.sentence_transformer import EmbeddingWithSentenceTransformer
from .load_arguments import ArgumentData

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalSubgraphExtractionInput(BaseModel):
    """
    MultimodalSubgraphExtractionInput is a Pydantic model representing an input for extracting a subgraph.

    Args:
        prompt: Prompt to interact with the backend.
        tool_call_id: Tool call ID.
        state: Injected state.
        arg_data: Argument for analytical process over graph data.
    """

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        description="Tool call ID."
    )
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")
    prompt: str = Field(description="Prompt to interact with the backend.")
    # list_of_genes: list = Field(
    #     description="List of genes to be used for subgraph extraction.", default=None
    # )
    # list_of_drugs: list = Field(
    #     description="List of drugs to be used for subgraph extraction.", default=None
    # )
    arg_data: ArgumentData = Field(
        description="Experiment over graph data.", default=None
    )


class MultimodalSubgraphExtractionTool(BaseTool):
    """
    This tool performs subgraph extraction based on user's prompt by taking into account
    the top-k nodes and edges.
    """

    name: str = "subgraph_extraction"
    description: str = "A tool for subgraph extraction based on user's prompt."
    args_schema: Type[BaseModel] = MultimodalSubgraphExtractionInput

    # def perform_endotype_filtering(
    #     self,
    #     prompt: str,
    #     state: Annotated[dict, InjectedState],
    #     cfg: hydra.core.config_store.ConfigStore,
    # ) -> str:
    #     """
    #     Perform endotype filtering based on the uploaded files and prepare the prompt.

    #     Args:
    #         prompt: The prompt to interact with the backend.
    #         state: Injected state for the tool.
    #         cfg: Hydra configuration object.
    #     """
    #     # Loop through the uploaded files
    #     all_genes = []
    #     for uploaded_file in state["uploaded_files"]:
    #         if uploaded_file["file_type"] == "endotype":
    #             # Load the PDF file
    #             docs = PyPDFLoader(file_path=uploaded_file["file_path"]).load()

    #             # Split the text into chunks
    #             splits = RecursiveCharacterTextSplitter(
    #                 chunk_size=cfg.splitter_chunk_size,
    #                 chunk_overlap=cfg.splitter_chunk_overlap,
    #             ).split_documents(docs)

    #             # Create a chat prompt template
    #             prompt_template = ChatPromptTemplate.from_messages(
    #                 [
    #                     ("system", cfg.prompt_endotype_filtering),
    #                     ("human", "{input}"),
    #                 ]
    #             )

    #             qa_chain = create_stuff_documents_chain(
    #                 state["llm_model"], prompt_template
    #             )
    #             rag_chain = create_retrieval_chain(
    #                 InMemoryVectorStore.from_documents(
    #                     documents=splits, embedding=state["embedding_model"]
    #                 ).as_retriever(
    #                     search_type=cfg.retriever_search_type,
    #                     search_kwargs={
    #                         "k": cfg.retriever_k,
    #                         "fetch_k": cfg.retriever_fetch_k,
    #                         "lambda_mult": cfg.retriever_lambda_mult,
    #                     },
    #                 ),
    #                 qa_chain,
    #             )
    #             results = rag_chain.invoke({"input": prompt})
    #             all_genes.append(results["answer"])

    #     # Prepare the prompt
    #     if len(all_genes) > 0:
    #         prompt = " ".join(
    #             [prompt, cfg.prompt_endotype_addition, ", ".join(all_genes)]
    #         )

    #     return prompt

    def prepare_final_subgraph(self,
                               state:Annotated[dict, InjectedState],
                               subgraph: dict,
                               pyg_graph: Data,
                               textualized_graph: pd.DataFrame) -> dict:
        """
        Prepare the subgraph based on the extracted subgraph.

        Args:
            state: The injected state for the tool.
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
            x=[pyg_graph.x[i] for i in subgraph["nodes"]],
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
            if n in state["selected_genes"]:
                nx_graph.add_node(n, color='#6a79f7')
            elif n in state["selected_drugs"]:
                nx_graph.add_node(n, color='#c4a661')
            else:
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
        # list_of_genes: list = None,
        arg_data: ArgumentData = None,
    ) -> Command:
        """
        Run the subgraph extraction tool.

        Args:
            tool_call_id: The tool call ID for the tool.
            state: Injected state for the tool.
            prompt: The prompt to interact with the backend.
            arg_data (ArgumentData): The argument data.

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
        initial_graph = {}
        initial_graph["source"] = state["dic_source_graph"][-1]  # The last source graph as of now
        # logger.log(logging.INFO, "Source graph: %s", source_graph)

        # Load the knowledge graph
        with open(initial_graph["source"]["kg_pyg_path"], "rb") as f:
            initial_graph["pyg"] = pickle.load(f)
        with open(initial_graph["source"]["kg_text_path"], "rb") as f:
            initial_graph["text"] = pickle.load(f)

        # Prepare prompt construction along with a list of endotypes
        # if len(state["uploaded_files"]) != 0 and "endotype" in [
        #     f["file_type"] for f in state["uploaded_files"]
        # ]:
        #     prompt = self.perform_endotype_filtering(prompt, state, cfg)

        # Prepare embeddings and modalities
        query_embs = []
        modalities = []

        # Text-based embeddings
        query_embs.append(
            torch.tensor(
                EmbeddingWithOllama(model_name=cfg.ollama_embeddings[0]).embed_query(prompt)
                # EmbeddingWithSentenceTransformer(
                #     model_name='nomic-ai/nomic-embed-text-v1.5',
                #     model_cache_dir="../../../../data/nomic-ai/nomic-embed-text-v1.5/",
                #     trust_remote_code=True).embed_query("search_query:" + prompt)
            ).float()
        )
        modalities.append("text")
        # Obtain another features from the selected genes
        for gene in state["selected_genes"]:
            query_embs.append(torch.tensor(
                initial_graph["pyg"].x[
                    np.where(np.array(initial_graph["pyg"].node_id) == gene)[0][0]
                ]
            ))
            modalities.append("sequence")
        # Obtain another features from the selected drugs
        for drug in state["selected_drugs"]:
            query_embs.append(torch.tensor(
                initial_graph["pyg"].x[
                    np.where(np.array(initial_graph["pyg"].node_id) == drug)[0][0]
                ]
            ))
            modalities.append("smiles")

        # Initialize the subgraph dictionary
        subgraphs = {}
        subgraphs["nodes"] = []
        subgraphs["edges"] = []

        # Loop over modalities and query embeddings
        for modality, query_emb in zip(modalities, query_embs):
            # Prepare the PCSTPruning object and extract the subgraph
            # Parameters were set in the configuration file obtained from Hydra
            subgraph = MultimodalPCSTPruning(                                         
                state["topk_nodes"],
                state["topk_edges"],
                cfg.cost_e,
                cfg.c_const,
                cfg.root,
                cfg.num_clusters,
                cfg.pruning,
                cfg.verbosity_level,
            ).extract_subgraph(initial_graph["pyg"],
                               query_embs[0], # use the first query embedding as the prompt
                               query_emb, 
                               modality)

            # Append the extracted subgraph to the dictionary
            subgraphs["nodes"].append(subgraph["nodes"].tolist())
            subgraphs["edges"].append(subgraph["edges"].tolist())

        # Concatenate and get unique node and edge indices
        subgraphs["nodes"] = np.unique(
            np.concatenate([np.array(list_) for list_ in subgraphs["nodes"]])
        )
        subgraphs["edges"] = np.unique(
            np.concatenate([np.array(list_) for list_ in subgraphs["edges"]])
        )

        # Prepare subgraph as a NetworkX graph and textualized graph
        final_subgraph = self.prepare_final_subgraph(
            state, subgraphs, initial_graph["pyg"], initial_graph["text"]
        )

        # Prepare the dictionary of extracted graph
        dic_extracted_graph = {
            "name": arg_data.extraction_name,
            "tool_call_id": tool_call_id,
            "graph_source": initial_graph["source"]["name"],
            "topk_nodes": state["topk_nodes"],
            "topk_edges": state["topk_edges"],
            "graph_dict": {
                "nodes": list(final_subgraph["graph_nx"].nodes(data=True)),
                "edges": list(final_subgraph["graph_nx"].edges(data=True)),
            },
            "graph_text": final_subgraph["graph_text"],
            "graph_summary": None,
        }

        # Prepare the dictionary of updated state
        dic_updated_state_for_model = {}
        for key, value in {
            "dic_extracted_graph": [dic_extracted_graph],
        }.items():
            if value:
                dic_updated_state_for_model[key] = value

        # Return the updated state of the tool
        return Command(
            update=dic_updated_state_for_model | {
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"Subgraph Extraction Result of {arg_data.extraction_name}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
