"""
Tool for performing multimodal subgraph extraction.
"""

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Annotated

import hydra
import pandas as pd
import pcst_fast
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from pymilvus import Collection

from ..utils.database import MilvusConnectionManager
from ..utils.database.milvus_connection_manager import QueryParams
from ..utils.extractions.milvus_multimodal_pcst import (
    DynamicLibraryLoader,
    MultimodalPCSTPruning,
    SystemDetector,
)
from .load_arguments import ArgumentData

# pylint: disable=too-many-lines
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionParams:
    """Parameters for subgraph extraction."""

    state: dict
    cfg: dict
    cfg_db: dict
    query_df: object
    connection_manager: object


class MultimodalSubgraphExtractionInput(BaseModel):
    """
    MultimodalSubgraphExtractionInput is a Pydantic model representing an input
    for extracting a subgraph.

    Args:
        prompt: Prompt to interact with the backend.
        tool_call_id: Tool call ID.
        state: Injected state.
        arg_data: Argument for analytical process over graph data.
    """

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(description="Tool call ID.")
    state: Annotated[dict, InjectedState] = Field(description="Injected state.")
    prompt: str = Field(description="Prompt to interact with the backend.")
    arg_data: ArgumentData = Field(description="Experiment over graph data.", default=None)


class MultimodalSubgraphExtractionTool(BaseTool):
    """
    This tool performs subgraph extraction based on user's prompt by taking into account
    the top-k nodes and edges.
    """

    name: str = "subgraph_extraction"
    description: str = "A tool for subgraph extraction based on user's prompt."
    args_schema: type[BaseModel] = MultimodalSubgraphExtractionInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize hardware detection and dynamic library loading
        object.__setattr__(self, "detector", SystemDetector())
        object.__setattr__(self, "loader", DynamicLibraryLoader(self.detector))
        logger.info(
            "MultimodalSubgraphExtractionTool initialized with %s mode",
            "GPU" if self.loader.use_gpu else "CPU",
        )

    def _read_multimodal_files(self, state: Annotated[dict, InjectedState]):
        """
        Read the uploaded multimodal files and return a DataFrame.

        Args:
            state: The injected state for the tool.

        Returns:
            A DataFrame containing the multimodal files.
        """
        multimodal_df = self.loader.df.DataFrame({"name": [], "node_type": []})

        # Loop over the uploaded files and find multimodal files
        logger.log(logging.INFO, "Looping over uploaded files")
        for i in range(len(state["uploaded_files"])):
            # Check if multimodal file is uploaded
            if state["uploaded_files"][i]["file_type"] == "multimodal":
                # Read the Excel file
                multimodal_df = pd.read_excel(
                    state["uploaded_files"][i]["file_path"], sheet_name=None
                )

        # Check if the multimodal_df is empty
        logger.log(logging.INFO, "Checking if multimodal_df is empty")
        if len(multimodal_df) > 0:
            # Prepare multimodal_df
            logger.log(logging.INFO, "Preparing multimodal_df")
            # Merge all obtained dataframes into a single dataframe
            multimodal_df = pd.concat(multimodal_df).reset_index()
            multimodal_df = self.loader.df.DataFrame(multimodal_df)
            multimodal_df.drop(columns=["level_1"], inplace=True)
            multimodal_df.rename(
                columns={"level_0": "q_node_type", "name": "q_node_name"}, inplace=True
            )
            # Since an excel sheet name could not contain a `/`,
            # but the node type can be 'gene/protein' as exists in the PrimeKG
            multimodal_df["q_node_type"] = multimodal_df["q_node_type"].str.replace("-", "_")

        return multimodal_df

    def _query_milvus_collection(self, node_type, node_type_df, cfg_db):
        """Helper method to query Milvus collection for a specific node type."""
        # Load the collection
        collection = Collection(
            name=f"{cfg_db.milvus_db.database_name}_nodes_{node_type.replace('/', '_')}"
        )
        collection.load()

        # Query the collection with node names from multimodal_df
        node_names_series = node_type_df["q_node_name"]
        q_node_names = getattr(
            node_names_series, "to_pandas", lambda series=node_names_series: series
        )().tolist()
        q_columns = [
            "node_id",
            "node_name",
            "node_type",
            "feat",
            "feat_emb",
            "desc",
            "desc_emb",
        ]
        res = collection.query(
            expr=f"node_name IN [{','.join(f'"{name}"' for name in q_node_names)}]",
            output_fields=q_columns,
        )
        # Convert the embeedings into floats
        for r_ in res:
            r_["feat_emb"] = [float(x) for x in r_["feat_emb"]]
            r_["desc_emb"] = [float(x) for x in r_["desc_emb"]]

        # Convert the result to a DataFrame
        res_df = self.loader.df.DataFrame(res)[q_columns]
        res_df["use_description"] = False
        return res_df

    async def _query_milvus_collection_async(
        self, node_type, node_type_df, cfg_db, connection_manager
    ):
        """Helper method to query Milvus collection asynchronously for a specific node type."""
        collection_name = f"{cfg_db.milvus_db.database_name}_nodes_{node_type.replace('/', '_')}"

        # Query the collection with node names from multimodal_df
        node_names_series = node_type_df["q_node_name"]
        q_node_names = getattr(
            node_names_series, "to_pandas", lambda series=node_names_series: series
        )().tolist()

        # Create filter expression for async query
        node_names_str = ",".join(f'"{name}"' for name in q_node_names)
        expr = f"node_name IN [{node_names_str}]"

        q_columns = [
            "node_id",
            "node_name",
            "node_type",
            "feat",
            "feat_emb",
            "desc",
            "desc_emb",
        ]

        # Create query parameters and perform async query
        query_params = QueryParams(
            collection_name=collection_name, expr=expr, output_fields=q_columns
        )
        res = await connection_manager.async_query(query_params)

        # Convert the embeddings into floats
        for r_ in res:
            r_["feat_emb"] = [float(x) for x in r_["feat_emb"]]
            r_["desc_emb"] = [float(x) for x in r_["desc_emb"]]

        # Convert the result to a DataFrame
        res_df = (
            self.loader.df.DataFrame(res)[q_columns]
            if res
            else self.loader.df.DataFrame(columns=q_columns)
        )
        res_df["use_description"] = False
        return res_df

    def _prepare_query_modalities(
        self, prompt: dict, state: Annotated[dict, InjectedState], cfg_db: dict
    ):
        """
        Prepare the modality-specific query for subgraph extraction.

        Args:
            prompt: The dictionary containing the user prompt and embeddings.
            state: The injected state for the tool.
            cfg_db: The configuration dictionary for Milvus database.

        Returns:
            A DataFrame containing the query embeddings and modalities.
        """
        # Initialize dataframes
        logger.log(logging.INFO, "Initializing dataframes")
        query_df = []
        prompt_df = self.loader.df.DataFrame(
            {
                "node_id": "user_prompt",
                "node_name": "User Prompt",
                "node_type": "prompt",
                "feat": prompt["text"],
                "feat_emb": prompt["emb"],
                "desc": prompt["text"],
                "desc_emb": prompt["emb"],
                "use_description": True,  # set to True for user prompt embedding
            }
        )

        # Read multimodal files uploaded by the user
        multimodal_df = self._read_multimodal_files(state)

        # Check if the multimodal_df is empty
        logger.log(logging.INFO, "Prepare query modalities")
        if len(multimodal_df) > 0:
            # Query the Milvus database for each node type in multimodal_df
            logger.log(
                logging.INFO,
                "Querying Milvus database for each node type in multimodal_df",
            )
            for node_type, node_type_df in multimodal_df.groupby("q_node_type"):
                print(f"Processing node type: {node_type}")
                res_df = self._query_milvus_collection(node_type, node_type_df, cfg_db)
                query_df.append(res_df)

            # Concatenate all results into a single DataFrame
            logger.log(logging.INFO, "Concatenating all results into a single DataFrame")
            query_df = self.loader.df.concat(query_df, ignore_index=True)

            # Update the state by adding the the selected node IDs
            logger.log(logging.INFO, "Updating state with selected node IDs")
            state["selections"] = (
                getattr(query_df, "to_pandas", lambda: query_df)()
                .groupby("node_type")["node_id"]
                .apply(list)
                .to_dict()
            )

            # Append a user prompt to the query dataframe
            logger.log(logging.INFO, "Adding user prompt to query dataframe")
            query_df = self.loader.df.concat([query_df, prompt_df]).reset_index(drop=True)
        else:
            # If no multimodal files are uploaded, use the prompt embeddings
            query_df = prompt_df

        return query_df

    async def _prepare_query_modalities_async(
        self,
        prompt: dict,
        state: Annotated[dict, InjectedState],
        cfg_db: dict,
        connection_manager,
    ):
        """
        Prepare the modality-specific query for subgraph extraction asynchronously.

        Args:
            prompt: The dictionary containing the user prompt and embeddings
            state: The injected state for the tool
            cfg_db: The configuration dictionary for Milvus database
            connection_manager: The MilvusConnectionManager instance

        Returns:
            A DataFrame containing the query embeddings and modalities
        """
        # Initialize dataframes
        logger.log(logging.INFO, "Initializing dataframes (async)")
        query_df = []
        prompt_df = self.loader.df.DataFrame(
            {
                "node_id": "user_prompt",
                "node_name": "User Prompt",
                "node_type": "prompt",
                "feat": prompt["text"],
                "feat_emb": prompt["emb"],
                "desc": prompt["text"],
                "desc_emb": prompt["emb"],
                "use_description": True,  # set to True for user prompt embedding
            }
        )

        # Read multimodal files uploaded by the user
        multimodal_df = self._read_multimodal_files(state)

        # Check if the multimodal_df is empty
        logger.log(logging.INFO, "Prepare query modalities (async)")
        if len(multimodal_df) > 0:
            # Create parallel tasks for querying each node type
            logger.log(
                logging.INFO,
                "Querying Milvus database for each node type in multimodal_df (parallel)",
            )

            # Create async tasks for each node type
            tasks = []
            for node_type, node_type_df in multimodal_df.groupby("q_node_type"):
                print(f"Processing node type: {node_type}")
                task = self._query_milvus_collection_async(
                    node_type, node_type_df, cfg_db, connection_manager
                )
                tasks.append(task)

            # Execute all queries in parallel using hybrid approach
            if len(tasks) == 1:
                # Single task, run directly
                query_results = [await tasks[0]]
            else:
                # Multiple tasks, but use sequential execution to avoid event loop issues
                query_results = []
                for task in tasks:
                    result = await task
                    query_results.append(result)

            query_df.extend(query_results)

            # Concatenate all results into a single DataFrame
            logger.log(logging.INFO, "Concatenating all results into a single DataFrame")
            query_df = self.loader.df.concat(query_df, ignore_index=True)

            # Update the state by adding the selected node IDs
            logger.log(logging.INFO, "Updating state with selected node IDs")
            state["selections"] = (
                getattr(query_df, "to_pandas", lambda: query_df)()
                .groupby("node_type")["node_id"]
                .apply(list)
                .to_dict()
            )

            # Append a user prompt to the query dataframe
            logger.log(logging.INFO, "Adding user prompt to query dataframe")
            query_df = self.loader.df.concat([query_df, prompt_df]).reset_index(drop=True)
        else:
            # If no multimodal files are uploaded, use the prompt embeddings
            query_df = prompt_df

        return query_df

    def _perform_subgraph_extraction(
        self,
        state: Annotated[dict, InjectedState],
        cfg: dict,
        cfg_db: dict,
        query_df: pd.DataFrame,
    ) -> dict:
        """
        Perform multimodal subgraph extraction based on modal-specific embeddings.

        Args:
            state: The injected state for the tool.
            cfg: The configuration dictionary.
            cfg_db: The configuration dictionary for Milvus database.
            query_df: The DataFrame containing the query embeddings and modalities.

        Returns:
            A dictionary containing the extracted subgraph with nodes and edges.
        """
        # Initialize the subgraph dictionary
        subgraphs = []
        unified_subgraph = {"nodes": [], "edges": []}
        # subgraphs = {}
        # subgraphs["nodes"] = []
        # subgraphs["edges"] = []

        # Loop over query embeddings and modalities
        for q in getattr(query_df, "to_pandas", lambda: query_df)().iterrows():
            logger.log(logging.INFO, "===========================================")
            logger.log(logging.INFO, "Processing query: %s", q[1]["node_name"])
            # Prepare the PCSTPruning object and extract the subgraph
            # Parameters were set in the configuration file obtained from Hydra
            # start = datetime.datetime.now()
            # Get dynamic metric type (overrides any config setting)
            # Get dynamic metric type (overrides any config setting)
            has_vector_processing = hasattr(cfg, "vector_processing")
            if has_vector_processing:
                dynamic_metrics_enabled = getattr(cfg.vector_processing, "dynamic_metrics", True)
            else:
                dynamic_metrics_enabled = False
            if has_vector_processing and dynamic_metrics_enabled:
                dynamic_metric_type = self.loader.metric_type
            else:
                dynamic_metric_type = getattr(cfg, "search_metric_type", self.loader.metric_type)

            subgraph = MultimodalPCSTPruning(
                topk=state["topk_nodes"],
                topk_e=state["topk_edges"],
                cost_e=cfg.cost_e,
                c_const=cfg.c_const,
                root=cfg.root,
                num_clusters=cfg.num_clusters,
                pruning=cfg.pruning,
                verbosity_level=cfg.verbosity_level,
                use_description=q[1]["use_description"],
                metric_type=dynamic_metric_type,  # Use dynamic or config metric type
                loader=self.loader,  # Pass the loader instance
            ).extract_subgraph(q[1]["desc_emb"], q[1]["feat_emb"], q[1]["node_type"], cfg_db)

            # Append the extracted subgraph to the dictionary
            unified_subgraph["nodes"].append(subgraph["nodes"].tolist())
            unified_subgraph["edges"].append(subgraph["edges"].tolist())
            subgraphs.append(
                (
                    q[1]["node_name"],
                    subgraph["nodes"].tolist(),
                    subgraph["edges"].tolist(),
                )
            )

            # end = datetime.datetime.now()
            # logger.log(logging.INFO, "Subgraph extraction time: %s seconds",
            #            (end - start).total_seconds())

        # Concatenate and get unique node and edge indices
        nodes_arrays = [self.loader.py.array(list_) for list_ in unified_subgraph["nodes"]]
        unified_subgraph["nodes"] = self.loader.py.unique(
            self.loader.py.concatenate(nodes_arrays)
        ).tolist()
        edges_arrays = [self.loader.py.array(list_) for list_ in unified_subgraph["edges"]]
        unified_subgraph["edges"] = self.loader.py.unique(
            self.loader.py.concatenate(edges_arrays)
        ).tolist()

        # Convert the unified subgraph and subgraphs to DataFrames
        unified_subgraph = self.loader.df.DataFrame(
            [
                (
                    "Unified Subgraph",
                    unified_subgraph["nodes"],
                    unified_subgraph["edges"],
                )
            ],
            columns=["name", "nodes", "edges"],
        )
        subgraphs = self.loader.df.DataFrame(subgraphs, columns=["name", "nodes", "edges"])

        # Concatenate both DataFrames
        subgraphs = self.loader.df.concat([unified_subgraph, subgraphs], ignore_index=True)

        return subgraphs

    async def _perform_subgraph_extraction_async(self, params: ExtractionParams) -> dict:
        """
        Perform multimodal subgraph extraction based on modal-specific embeddings asynchronously.

        Args:
            state: The injected state for the tool
            cfg: The configuration dictionary
            cfg_db: The configuration dictionary for Milvus database
            query_df: The DataFrame containing the query embeddings and modalities
            connection_manager: The MilvusConnectionManager instance

        Returns:
            A dictionary containing the extracted subgraph with nodes and edges
        """
        # Initialize the subgraph dictionary
        subgraphs = []
        unified_subgraph = {"nodes": [], "edges": []}

        # Create parallel tasks for each query
        tasks = []
        query_info = []

        for q in getattr(params.query_df, "to_pandas", lambda: params.query_df)().iterrows():
            logger.log(logging.INFO, "===========================================")
            logger.log(logging.INFO, "Processing query: %s", q[1]["node_name"])

            # Store query info for later processing
            query_info.append(q[1])

            # Get dynamic metric type using helper method
            dynamic_metric_type = self._get_dynamic_metric_type(params.cfg)

            # Create PCST pruning instance using helper
            pcst_instance = self._create_pcst_instance(params, q[1], dynamic_metric_type)

            # Create async task for subgraph extraction
            task = self._extract_single_subgraph_async(
                pcst_instance, q[1], params.cfg_db, params.connection_manager
            )
            tasks.append(task)

        # Execute all subgraph extractions sequentially to avoid event loop conflicts
        subgraph_results = []
        for i, task in enumerate(tasks):
            logger.log(logging.INFO, "Processing subgraph %d/%d", i + 1, len(tasks))
            result = await task
            subgraph_results.append(result)

        # Process results and finalize
        self._process_subgraph_results(subgraph_results, query_info, unified_subgraph, subgraphs)
        return self._finalize_subgraph_results(subgraphs, unified_subgraph)

    def _process_subgraph_results(self, subgraph_results, query_info, unified_subgraph, subgraphs):
        """Process individual subgraph results."""
        for i, subgraph in enumerate(subgraph_results):
            query_row = query_info[i]
            unified_subgraph["nodes"].append(subgraph["nodes"].tolist())
            unified_subgraph["edges"].append(subgraph["edges"].tolist())
            subgraphs.append(
                (
                    query_row["node_name"],
                    subgraph["nodes"].tolist(),
                    subgraph["edges"].tolist(),
                )
            )

    def _finalize_subgraph_results(self, subgraphs, unified_subgraph):
        """Process and finalize subgraph results into DataFrames."""
        # Concatenate and get unique node and edge indices
        nodes_arrays = [self.loader.py.array(list_) for list_ in unified_subgraph["nodes"]]
        unified_subgraph["nodes"] = self.loader.py.unique(
            self.loader.py.concatenate(nodes_arrays)
        ).tolist()
        edges_arrays = [self.loader.py.array(list_) for list_ in unified_subgraph["edges"]]
        unified_subgraph["edges"] = self.loader.py.unique(
            self.loader.py.concatenate(edges_arrays)
        ).tolist()

        # Convert the unified subgraph and subgraphs to DataFrames
        unified_subgraph_df = self.loader.df.DataFrame(
            [
                (
                    "Unified Subgraph",
                    unified_subgraph["nodes"],
                    unified_subgraph["edges"],
                )
            ],
            columns=["name", "nodes", "edges"],
        )
        subgraphs_df = self.loader.df.DataFrame(subgraphs, columns=["name", "nodes", "edges"])

        # Concatenate both DataFrames
        return self.loader.df.concat([unified_subgraph_df, subgraphs_df], ignore_index=True)

    async def _extract_single_subgraph_async(
        self, pcst_instance, query_row, cfg_db, connection_manager
    ):
        """
        Extract a single subgraph asynchronously using the new async methods.
        """
        # Load data and compute prizes
        edge_index, prizes, num_nodes = await self._load_subgraph_data(
            pcst_instance, query_row, cfg_db, connection_manager
        )

        # Run PCST algorithm and get results
        return self._run_pcst_algorithm(pcst_instance, edge_index, num_nodes, prizes)

    async def _load_subgraph_data(self, pcst_instance, query_row, cfg_db, connection_manager):
        """Load edge index, compute prizes, and get node count."""
        # Load edge index asynchronously
        edge_index = await pcst_instance.load_edge_index_async(cfg_db, connection_manager)

        # Compute prizes asynchronously
        prizes = await pcst_instance.compute_prizes_async(
            query_row["desc_emb"],
            query_row["feat_emb"],
            cfg_db,
            query_row["node_type"],
        )

        # Get number of nodes
        nodes_collection = f"{cfg_db.milvus_db.database_name}_nodes"
        stats = await connection_manager.async_get_collection_stats(nodes_collection)
        num_nodes = stats["num_entities"]

        return edge_index, prizes, num_nodes

    def _run_pcst_algorithm(self, pcst_instance, edge_index, num_nodes, prizes):
        """Run PCST algorithm and get subgraph results."""
        # Compute costs in constructing the subgraph
        edges_dict, prizes_final, costs, mapping = pcst_instance.compute_subgraph_costs(
            edge_index, num_nodes, prizes
        )

        # Retrieve the subgraph using the PCST algorithm
        result_vertices, result_edges = pcst_fast.pcst_fast(
            edges_dict["edges"].tolist(),
            prizes_final.tolist(),
            costs.tolist(),
            pcst_instance.root,
            pcst_instance.num_clusters,
            pcst_instance.pruning,
            pcst_instance.verbosity_level,
        )

        # Get subgraph nodes and edges based on the PCST result
        return pcst_instance.get_subgraph_nodes_edges(
            num_nodes,
            pcst_instance.loader.py.asarray(result_vertices),
            {
                "edges": pcst_instance.loader.py.asarray(result_edges),
                "num_prior_edges": edges_dict["num_prior_edges"],
                "edge_index": edge_index,
            },
            mapping,
        )

    def _run(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
        arg_data: ArgumentData = None,
    ) -> Command:
        """
        Synchronous wrapper for the async _run_async method.
        This maintains compatibility with LangGraph while using async operations internally.
        """
        # concurrent.futures imported at top level

        def run_in_thread():
            """Run async method in a new thread with its own event loop."""
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(
                    self._run_async(tool_call_id, state, prompt, arg_data)
                )
                return result
            finally:
                # Properly cleanup the event loop
                new_loop.close()
                asyncio.set_event_loop(None)

        # Always use a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_thread)
            return future.result()

    def _prepare_final_subgraph(
        self, state: Annotated[dict, InjectedState], subgraph: dict, cfg_db
    ) -> dict:
        """
        Prepare the subgraph based on the extracted subgraph.

        Args:
            state: The injected state for the tool.
            subgraph: The extracted subgraph.
            cfg_db: The configuration dictionary for Milvus database.

        Returns:
            A dictionary containing the PyG graph, NetworkX graph, and textualized graph.
        """
        # Convert the dict to a DataFrame
        node_colors = {
            n: cfg_db.node_colors_dict[k] for k, v in state["selections"].items() for n in v
        }
        color_df = self.loader.df.DataFrame(list(node_colors.items()), columns=["node_id", "color"])
        # print(color_df)

        # Prepare the subgraph dictionary
        graph_dict = {"name": [], "nodes": [], "edges": [], "text": ""}
        for sub in getattr(subgraph, "to_pandas", lambda: subgraph)().itertuples(index=False):
            graph_nodes, graph_edges = self._process_subgraph_data(sub, cfg_db, color_df)

            # Prepare lists for visualization
            graph_dict["name"].append(sub.name)
            graph_dict["nodes"].append(
                [
                    (
                        row.node_id,
                        {
                            "hover": "Node Name : "
                            + row.node_name
                            + "\n"
                            + "Node Type : "
                            + row.node_type
                            + "\n"
                            + "Desc : "
                            + row.desc,
                            "click": "$hover",
                            "color": row.color,
                        },
                    )
                    for row in getattr(
                        graph_nodes,
                        "to_pandas",
                        lambda graph_nodes=graph_nodes: graph_nodes,
                    )().itertuples(index=False)
                ]
            )
            graph_dict["edges"].append(
                [
                    (row.head_id, row.tail_id, {"label": tuple(row.edge_type)})
                    for row in getattr(
                        graph_edges,
                        "to_pandas",
                        lambda graph_edges=graph_edges: graph_edges,
                    )().itertuples(index=False)
                ]
            )

            # Prepare the textualized subgraph
            if sub.name == "Unified Subgraph":
                graph_nodes = graph_nodes[["node_id", "desc"]]
                graph_nodes.rename(columns={"desc": "node_attr"}, inplace=True)
                graph_edges = graph_edges[["head_id", "edge_type", "tail_id"]]
                nodes_pandas = getattr(
                    graph_nodes,
                    "to_pandas",
                    lambda graph_nodes=graph_nodes: graph_nodes,
                )()
                nodes_csv = nodes_pandas.to_csv(index=False)
                edges_pandas = getattr(
                    graph_edges,
                    "to_pandas",
                    lambda graph_edges=graph_edges: graph_edges,
                )()
                edges_csv = edges_pandas.to_csv(index=False)
                graph_dict["text"] = nodes_csv + "\n" + edges_csv

        return graph_dict

    def _process_subgraph_data(self, sub, cfg_db, color_df):
        """Helper method to process individual subgraph data."""
        print(f"Processing subgraph: {sub.name}")
        print("---")
        print(sub.nodes)
        print("---")
        print(sub.edges)
        print("---")

        # Prepare graph dataframes - Nodes
        coll_name = f"{cfg_db.milvus_db.database_name}_nodes"
        node_coll = Collection(name=coll_name)
        node_coll.load()
        graph_nodes = node_coll.query(
            expr=f"node_index IN [{','.join(f'{n}' for n in sub.nodes)}]",
            output_fields=["node_id", "node_name", "node_type", "desc"],
        )
        graph_nodes = self.loader.df.DataFrame(graph_nodes)
        graph_nodes.drop(columns=["node_index"], inplace=True)
        if not color_df.empty:
            graph_nodes = graph_nodes.merge(color_df, on="node_id", how="left")
        else:
            graph_nodes["color"] = "black"
        graph_nodes["color"] = graph_nodes["color"].fillna("black")

        # Edges
        coll_name = f"{cfg_db.milvus_db.database_name}_edges"
        edge_coll = Collection(name=coll_name)
        edge_coll.load()
        graph_edges = edge_coll.query(
            expr=f"triplet_index IN [{','.join(f'{e}' for e in sub.edges)}]",
            output_fields=["head_id", "tail_id", "edge_type"],
        )
        graph_edges = self.loader.df.DataFrame(graph_edges)
        graph_edges.drop(columns=["triplet_index"], inplace=True)
        graph_edges["edge_type"] = graph_edges["edge_type"].str.split("|")

        return graph_nodes, graph_edges

    def _get_dynamic_metric_type(self, cfg: dict) -> str:
        """Helper method to get dynamic metric type."""
        has_vector_processing = hasattr(cfg, "vector_processing")
        if has_vector_processing:
            dynamic_metrics_enabled = getattr(cfg.vector_processing, "dynamic_metrics", True)
        else:
            dynamic_metrics_enabled = False
        if has_vector_processing and dynamic_metrics_enabled:
            return self.loader.metric_type
        return getattr(cfg, "search_metric_type", self.loader.metric_type)

    def _create_pcst_instance(
        self, params: ExtractionParams, query_row: dict, dynamic_metric_type: str
    ) -> MultimodalPCSTPruning:
        """Helper method to create PCST pruning instance."""
        return MultimodalPCSTPruning(
            topk=params.state["topk_nodes"],
            topk_e=params.state["topk_edges"],
            cost_e=params.cfg.cost_e,
            c_const=params.cfg.c_const,
            root=params.cfg.root,
            num_clusters=params.cfg.num_clusters,
            pruning=params.cfg.pruning,
            verbosity_level=params.cfg.verbosity_level,
            use_description=query_row["use_description"],
            metric_type=dynamic_metric_type,
            loader=self.loader,
        )

    def normalize_vector(self, v: list) -> list:
        """
        Normalize a vector using appropriate library (CuPy for GPU, NumPy for CPU).

        Args:
            v : Vector to normalize.

        Returns:
            Normalized vector.
        """
        if self.loader.normalize_vectors:
            # GPU mode: normalize the vector
            v_array = self.loader.py.asarray(v)
            norm = self.loader.py.linalg.norm(v_array)
            return (v_array / norm).tolist()
        # CPU mode: return as-is for COSINE similarity
        return v

    async def _run_async(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        prompt: str,
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
                config_name="config",
                overrides=["tools/multimodal_subgraph_extraction=default"],
            )
            cfg = cfg.tools.multimodal_subgraph_extraction

        # Load database configuration separately
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg_all = hydra.compose(config_name="config")
            cfg_db = cfg_all.utils.database.milvus

        # Establish Milvus connection using singleton connection manager
        logger.log(logging.INFO, "Getting Milvus connection manager (singleton)")
        connection_manager = MilvusConnectionManager(cfg_db)
        try:
            connection_manager.ensure_connection()
            logger.log(logging.INFO, "Milvus connection established successfully")

            # Log connection info
            conn_info = connection_manager.get_connection_info()
            logger.log(logging.INFO, "Connected to database: %s", conn_info.get("database"))
            logger.log(
                logging.INFO,
                "Connection healthy: %s",
                connection_manager.test_connection(),
            )
        except Exception as e:
            logger.error("Failed to establish Milvus connection: %s", str(e))
            raise RuntimeError(f"Cannot connect to Milvus database: {str(e)}") from e

        # Prepare the query embeddings and modalities (async)
        logger.log(logging.INFO, "_prepare_query_modalities_async")
        query_df = await self._prepare_query_modalities_async(
            {
                "text": prompt,
                "emb": [self.normalize_vector(state["embedding_model"].embed_query(prompt))],
            },
            state,
            cfg_db,
            connection_manager,
        )

        # Perform subgraph extraction (async)
        logger.log(logging.INFO, "_perform_subgraph_extraction_async")
        extraction_params = ExtractionParams(
            state=state,
            cfg=cfg,
            cfg_db=cfg_db,
            query_df=query_df,
            connection_manager=connection_manager,
        )
        subgraphs = await self._perform_subgraph_extraction_async(extraction_params)

        # Prepare subgraph as a NetworkX graph and textualized graph
        logger.log(logging.INFO, "_prepare_final_subgraph")
        logger.log(logging.INFO, "Subgraphs extracted: %s", len(subgraphs))
        # start = datetime.datetime.now()
        final_subgraph = self._prepare_final_subgraph(state, subgraphs, cfg_db)
        # end = datetime.datetime.now()
        # logger.log(logging.INFO, "_prepare_final_subgraph time: %s seconds",
        #            (end - start).total_seconds())

        # Create final result and return command
        return self._create_extraction_result(tool_call_id, state, final_subgraph, arg_data)

    def _create_extraction_result(self, tool_call_id, state, final_subgraph, arg_data):
        """Create the final extraction result and command."""
        # Prepare the dictionary of extracted graph
        logger.log(logging.INFO, "dic_extracted_graph")
        dic_extracted_graph = {
            "name": arg_data.extraction_name,
            "tool_call_id": tool_call_id,
            "graph_source": state["dic_source_graph"][0]["name"],
            "topk_nodes": state["topk_nodes"],
            "topk_edges": state["topk_edges"],
            "graph_dict": {
                "name": final_subgraph["name"],
                "nodes": final_subgraph["nodes"],
                "edges": final_subgraph["edges"],
            },
            "graph_text": final_subgraph["text"],
            "graph_summary": None,
        }

        # Debug logging
        logger.info(
            "Created dic_extracted_graph with keys: %s",
            list(dic_extracted_graph.keys()),
        )
        logger.info(
            "Graph dict structure - name count: %d, nodes count: %d, edges count: %d",
            len(dic_extracted_graph["graph_dict"]["name"]),
            len(dic_extracted_graph["graph_dict"]["nodes"]),
            len(dic_extracted_graph["graph_dict"]["edges"]),
        )

        # Create success message
        success_message = (
            f"Successfully extracted subgraph '{arg_data.extraction_name}' "
            f"with {len(final_subgraph['name'])} graph(s). The subgraph contains "
            f"{sum(len(nodes) for nodes in final_subgraph['nodes'])} nodes and "
            f"{sum(len(edges) for edges in final_subgraph['edges'])} edges. "
            "The extracted subgraph has been stored and is ready for "
            "visualization and analysis."
        )

        # Return the command with updated state
        return Command(
            update={"dic_extracted_graph": [dic_extracted_graph]}
            | {
                "messages": [ToolMessage(content=success_message, tool_call_id=tool_call_id)],
            }
        )
