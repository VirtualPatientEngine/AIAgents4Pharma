"""
Test cases for tools/milvus_multimodal_subgraph_extraction.py
"""

import importlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ..tools.milvus_multimodal_subgraph_extraction import MultimodalSubgraphExtractionTool


class TestMultimodalSubgraphExtractionTool(unittest.TestCase):
    """
    Test cases for MultimodalSubgraphExtractionTool (Milvus)
    """

    def setUp(self):
        self.tool = MultimodalSubgraphExtractionTool()
        self.state = {
            "uploaded_files": [],
            "embedding_model": MagicMock(),
            "topk_nodes": 5,
            "topk_edges": 5,
            "dic_source_graph": [{"name": "TestGraph"}],
        }
        self.prompt = "Find subgraph for test"
        self.arg_data = {"extraction_name": "subkg_12345"}
        self.cfg_db = MagicMock()
        self.cfg_db.milvus_db.database_name = "testdb"
        self.cfg_db.milvus_db.alias = "default"
        self.cfg = MagicMock()
        self.cfg.cost_e = 1.0
        self.cfg.c_const = 1.0
        self.cfg.root = 0
        self.cfg.num_clusters = 1
        self.cfg.pruning = True
        self.cfg.verbosity_level = 0
        self.cfg.search_metric_type = "L2"
        self.cfg.node_colors_dict = {"gene/protein": "red"}

    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.Collection"
    )
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.MultimodalPCSTPruning"
    )
    @patch("pymilvus.connections")
    def test_extract_multimodal_subgraph_wo_doc(self, mock_connections, mock_pcst, mock_collection):
        """
        Test the multimodal subgraph extraction tool for only text as modality.
        """

        # Mock Milvus connection utilities
        mock_connections.has_connection.return_value = True

        # No uploaded_files (no doc)
        self.state["uploaded_files"] = []
        self.state["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
        self.state["selections"] = {}

        # Mock Collection for nodes and edges
        colls = {}
        colls["nodes"] = MagicMock()
        colls["nodes"] = MagicMock()
        colls["nodes"].query.return_value = [
            {
                "node_index": 0,
                "node_id": "id1",
                "node_name": "JAK1",
                "node_type": "gene/protein",
                "feat": "featA",
                "feat_emb": [0.1, 0.2, 0.3],
                "desc": "descA",
                "desc_emb": [0.1, 0.2, 0.3],
            },
            {
                "node_index": 1,
                "node_id": "id2",
                "node_name": "JAK2",
                "node_type": "gene/protein",
                "feat": "featB",
                "feat_emb": [0.4, 0.5, 0.6],
                "desc": "descB",
                "desc_emb": [0.4, 0.5, 0.6],
            },
        ]
        colls["nodes"].load.return_value = None

        colls["edges"] = MagicMock()
        colls["edges"].query.return_value = [
            {
                "triplet_index": 0,
                "head_id": "id1",
                "head_index": 0,
                "tail_id": "id2",
                "tail_index": 1,
                "edge_type": "gene/protein,ppi,gene/protein",
                "display_relation": "ppi",
                "feat": "featC",
                "feat_emb": [0.7, 0.8, 0.9],
            }
        ]
        colls["edges"].load.return_value = None

        def collection_side_effect(name):
            """
            Mock side effect for Collection to return nodes or edges based on name.
            """
            if "nodes" in name:
                return colls["nodes"]
            if "edges" in name:
                return colls["edges"]
            return None

        mock_collection.side_effect = collection_side_effect

        # Mock MultimodalPCSTPruning
        mock_pcst_instance = MagicMock()
        mock_pcst_instance.extract_subgraph.return_value = {
            "nodes": pd.Series([1, 2]),
            "edges": pd.Series([0]),
        }
        mock_pcst.return_value = mock_pcst_instance

        # Patch hydra.compose to return config objects
        with (
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.initialize"
            ),
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.compose"
            ) as mock_compose,
        ):
            mock_compose.return_value = MagicMock()
            mock_compose.return_value.app.frontend = self.cfg_db
            mock_compose.return_value.tools.multimodal_subgraph_extraction = self.cfg

            response = self.tool.invoke(
                input={
                    "prompt": self.prompt,
                    "tool_call_id": "subgraph_extraction_tool",
                    "state": self.state,
                    "arg_data": self.arg_data,
                }
            )

        # Check tool message
        self.assertEqual(response.update["messages"][-1].tool_call_id, "subgraph_extraction_tool")

        # Check extracted subgraph dictionary
        dic_extracted_graph = response.update["dic_extracted_graph"][0]
        self.assertIsInstance(dic_extracted_graph, dict)
        self.assertEqual(dic_extracted_graph["name"], self.arg_data["extraction_name"])
        self.assertEqual(dic_extracted_graph["graph_source"], "TestGraph")
        self.assertEqual(dic_extracted_graph["topk_nodes"], 5)
        self.assertEqual(dic_extracted_graph["topk_edges"], 5)
        self.assertIsInstance(dic_extracted_graph["graph_dict"], dict)
        self.assertGreater(len(dic_extracted_graph["graph_dict"]["nodes"]), 0)
        self.assertGreater(len(dic_extracted_graph["graph_dict"]["edges"]), 0)
        self.assertIsInstance(dic_extracted_graph["graph_text"], str)
        # Check if the nodes are in the graph_text
        self.assertTrue(
            all(
                n[0] in dic_extracted_graph["graph_text"].replace('"', "")
                for subgraph_nodes in dic_extracted_graph["graph_dict"]["nodes"]
                for n in subgraph_nodes
            )
        )
        # Check if the edges are in the graph_text
        self.assertTrue(
            all(
                ",".join([str(e[0])] + str(e[2]["label"][0]).split(",") + [str(e[1])])
                in dic_extracted_graph["graph_text"]
                .replace('"', "")
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                for subgraph_edges in dic_extracted_graph["graph_dict"]["edges"]
                for e in subgraph_edges
            )
        )

        # Another test for unknown collection
        result = collection_side_effect("unknown")
        self.assertIsNone(result)

    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.Collection"
    )
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.pd.read_excel"
    )
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.MultimodalPCSTPruning"
    )
    @patch("pymilvus.connections")
    def test_extract_multimodal_subgraph_w_doc(
        self, mock_connections, mock_pcst, mock_read_excel, mock_collection
    ):
        """
        Test the multimodal subgraph extraction tool for text as modality, plus genes.
        """
        # Mock Milvus connection utilities
        mock_connections.has_connection.return_value = True

        # With uploaded_files (with doc)
        self.state["uploaded_files"] = [{"file_type": "multimodal", "file_path": "dummy.xlsx"}]
        self.state["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
        self.state["selections"] = {"gene/protein": ["JAK1", "JAK2"]}

        # Mock pd.read_excel to return a dict of DataFrames
        df = pd.DataFrame({"name": ["JAK1", "JAK2"], "node_type": ["gene/protein", "gene/protein"]})
        mock_read_excel.return_value = {"gene/protein": df}

        # Mock Collection for nodes and edges
        colls = {}
        colls["nodes"] = MagicMock()
        colls["nodes"] = MagicMock()
        colls["nodes"].query.return_value = [
            {
                "node_index": 0,
                "node_id": "id1",
                "node_name": "JAK1",
                "node_type": "gene/protein",
                "feat": "featA",
                "feat_emb": [0.1, 0.2, 0.3],
                "desc": "descA",
                "desc_emb": [0.1, 0.2, 0.3],
            },
            {
                "node_index": 1,
                "node_id": "id2",
                "node_name": "JAK2",
                "node_type": "gene/protein",
                "feat": "featB",
                "feat_emb": [0.4, 0.5, 0.6],
                "desc": "descB",
                "desc_emb": [0.4, 0.5, 0.6],
            },
        ]
        colls["nodes"].load.return_value = None

        colls["edges"] = MagicMock()
        colls["edges"].query.return_value = [
            {
                "triplet_index": 0,
                "head_id": "id1",
                "head_index": 0,
                "tail_id": "id2",
                "tail_index": 1,
                "edge_type": "gene/protein,ppi,gene/protein",
                "display_relation": "ppi",
                "feat": "featC",
                "feat_emb": [0.7, 0.8, 0.9],
            }
        ]
        colls["edges"].load.return_value = None

        def collection_side_effect(name):
            """
            Mock side effect for Collection to return nodes or edges based on name.
            """
            if "nodes" in name:
                return colls["nodes"]
            if "edges" in name:
                return colls["edges"]
            return None

        mock_collection.side_effect = collection_side_effect

        # Mock MultimodalPCSTPruning
        mock_pcst_instance = MagicMock()
        mock_pcst_instance.extract_subgraph.return_value = {
            "nodes": pd.Series([1, 2]),
            "edges": pd.Series([0]),
        }
        mock_pcst.return_value = mock_pcst_instance

        # Patch hydra.compose to return config objects
        with (
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.initialize"
            ),
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.compose"
            ) as mock_compose,
        ):
            mock_compose.return_value = MagicMock()
            mock_compose.return_value.app.frontend = self.cfg_db
            mock_compose.return_value.tools.multimodal_subgraph_extraction = self.cfg

            response = self.tool.invoke(
                input={
                    "prompt": self.prompt,
                    "tool_call_id": "subgraph_extraction_tool",
                    "state": self.state,
                    "arg_data": self.arg_data,
                }
            )

        # Check tool message
        self.assertEqual(response.update["messages"][-1].tool_call_id, "subgraph_extraction_tool")

        # Check extracted subgraph dictionary
        dic_extracted_graph = response.update["dic_extracted_graph"][0]
        self.assertIsInstance(dic_extracted_graph, dict)
        self.assertEqual(dic_extracted_graph["name"], self.arg_data["extraction_name"])
        self.assertEqual(dic_extracted_graph["graph_source"], "TestGraph")
        self.assertEqual(dic_extracted_graph["topk_nodes"], 5)
        self.assertEqual(dic_extracted_graph["topk_edges"], 5)
        self.assertIsInstance(dic_extracted_graph["graph_dict"], dict)
        self.assertGreater(len(dic_extracted_graph["graph_dict"]["nodes"]), 0)
        self.assertGreater(len(dic_extracted_graph["graph_dict"]["edges"]), 0)
        self.assertIsInstance(dic_extracted_graph["graph_text"], str)
        # Check if the nodes are in the graph_text
        self.assertTrue(
            all(
                n[0] in dic_extracted_graph["graph_text"].replace('"', "")
                for subgraph_nodes in dic_extracted_graph["graph_dict"]["nodes"]
                for n in subgraph_nodes
            )
        )
        # Check if the edges are in the graph_text
        self.assertTrue(
            all(
                ",".join([str(e[0])] + str(e[2]["label"][0]).split(",") + [str(e[1])])
                in dic_extracted_graph["graph_text"]
                .replace('"', "")
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                for subgraph_edges in dic_extracted_graph["graph_dict"]["edges"]
                for e in subgraph_edges
            )
        )

        # Another test for unknown collection
        result = collection_side_effect("unknown")
        self.assertIsNone(result)

    def test_extract_multimodal_subgraph_wo_doc_gpu(self):
        """
        Test the multimodal subgraph extraction tool for only text as modality,
        simulating GPU (cudf/cupy) environment.
        """
        module_name = (
            "aiagents4pharma.talk2knowledgegraphs.tools." + "milvus_multimodal_subgraph_extraction"
        )
        with patch.dict("sys.modules", {"cupy": np, "cudf": pd}):
            mod = importlib.reload(importlib.import_module(module_name))
            # Patch Collection and MultimodalPCSTPruning after reload
            with (
                patch(f"{module_name}.Collection") as mock_collection,
                patch(f"{module_name}.MultimodalPCSTPruning") as mock_pcst,
                patch("pymilvus.connections") as mock_connections,
            ):
                # Setup mocks as in the original test
                mock_connections.has_connection.return_value = True
                colls = {}
                colls["nodes"] = MagicMock()
                colls["nodes"].query.return_value = [
                    {
                        "node_index": 0,
                        "node_id": "id1",
                        "node_name": "JAK1",
                        "node_type": "gene/protein",
                        "feat": "featA",
                        "feat_emb": [0.1, 0.2, 0.3],
                        "desc": "descA",
                        "desc_emb": [0.1, 0.2, 0.3],
                    },
                    {
                        "node_index": 1,
                        "node_id": "id2",
                        "node_name": "JAK2",
                        "node_type": "gene/protein",
                        "feat": "featB",
                        "feat_emb": [0.4, 0.5, 0.6],
                        "desc": "descB",
                        "desc_emb": [0.4, 0.5, 0.6],
                    },
                ]
                colls["nodes"].load.return_value = None
                colls["edges"] = MagicMock()
                colls["edges"].query.return_value = [
                    {
                        "triplet_index": 0,
                        "head_id": "id1",
                        "head_index": 0,
                        "tail_id": "id2",
                        "tail_index": 1,
                        "edge_type": "gene/protein,ppi,gene/protein",
                        "display_relation": "ppi",
                        "feat": "featC",
                        "feat_emb": [0.7, 0.8, 0.9],
                    }
                ]
                colls["edges"].load.return_value = None

                def collection_side_effect(name):
                    if "nodes" in name:
                        return colls["nodes"]
                    if "edges" in name:
                        return colls["edges"]
                    return None

                mock_collection.side_effect = collection_side_effect
                mock_pcst_instance = MagicMock()
                mock_pcst_instance.extract_subgraph.return_value = {
                    "nodes": pd.Series([1, 2]),
                    "edges": pd.Series([0]),
                }
                mock_pcst.return_value = mock_pcst_instance
                # Setup config mocks
                tool_cls = mod.MultimodalSubgraphExtractionTool
                tool = tool_cls()

                # Patch hydra.compose
                with (
                    patch(f"{module_name}.hydra.initialize"),
                    patch(f"{module_name}.hydra.compose") as mock_compose,
                ):
                    mock_compose.return_value = MagicMock()
                    mock_compose.return_value.app.frontend = self.cfg_db
                    mock_compose.return_value.tools.multimodal_subgraph_extraction = self.cfg
                    self.state["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
                    self.state["selections"] = {}
                    response = tool.invoke(
                        input={
                            "prompt": self.prompt,
                            "tool_call_id": "subgraph_extraction_tool",
                            "state": self.state,
                            "arg_data": self.arg_data,
                        }
                    )
                # Check tool message
                self.assertEqual(
                    response.update["messages"][-1].tool_call_id, "subgraph_extraction_tool"
                )
                dic_extracted_graph = response.update["dic_extracted_graph"][0]
                self.assertIsInstance(dic_extracted_graph, dict)
                self.assertEqual(dic_extracted_graph["name"], self.arg_data["extraction_name"])
                self.assertEqual(dic_extracted_graph["graph_source"], "TestGraph")
                self.assertEqual(dic_extracted_graph["topk_nodes"], 5)
                self.assertEqual(dic_extracted_graph["topk_edges"], 5)
                self.assertIsInstance(dic_extracted_graph["graph_dict"], dict)
                self.assertGreater(len(dic_extracted_graph["graph_dict"]["nodes"]), 0)
                self.assertGreater(len(dic_extracted_graph["graph_dict"]["edges"]), 0)
                self.assertIsInstance(dic_extracted_graph["graph_text"], str)
                self.assertTrue(
                    all(
                        n[0] in dic_extracted_graph["graph_text"].replace('"', "")
                        for subgraph_nodes in dic_extracted_graph["graph_dict"]["nodes"]
                        for n in subgraph_nodes
                    )
                )
                self.assertTrue(
                    all(
                        ",".join([str(e[0])] + str(e[2]["label"][0]).split(",") + [str(e[1])])
                        in dic_extracted_graph["graph_text"]
                        .replace('"', "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        for subgraph_edges in dic_extracted_graph["graph_dict"]["edges"]
                        for e in subgraph_edges
                    )
                )

                # Another test for unknown collection
                result = collection_side_effect("unknown")
                self.assertIsNone(result)

    def test_normalize_vector_gpu_mode(self):
        """Test normalize_vector method in GPU mode."""
        # Mock the loader to simulate GPU mode
        self.tool.loader.normalize_vectors = True
        self.tool.loader.py = MagicMock()
        # Mock the GPU array operations
        mock_array = MagicMock()
        mock_norm = MagicMock()
        mock_norm.return_value = 2.0
        mock_array.__truediv__ = MagicMock(return_value=mock_array)
        mock_array.tolist.return_value = [0.5, 1.0, 1.5]
        self.tool.loader.py.asarray.return_value = mock_array
        self.tool.loader.py.linalg.norm.return_value = mock_norm
        result = self.tool.normalize_vector([1.0, 2.0, 3.0])
        # Verify the result
        self.assertEqual(result, [0.5, 1.0, 1.5])
        self.tool.loader.py.asarray.assert_called_once_with([1.0, 2.0, 3.0])
        self.tool.loader.py.linalg.norm.assert_called_once_with(mock_array)

    def test_normalize_vector_cpu_mode(self):
        """Test normalize_vector method in CPU mode."""
        # Mock the loader to simulate CPU mode
        self.tool.loader.normalize_vectors = False
        result = self.tool.normalize_vector([1.0, 2.0, 3.0])
        # In CPU mode, should return the input as-is
        self.assertEqual(result, [1.0, 2.0, 3.0])

    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.Collection"
    )
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.MultimodalPCSTPruning"
    )
    @patch("pymilvus.connections")
    def test_extract_multimodal_subgraph_no_vector_processing(
        self, mock_connections, mock_pcst, mock_collection
    ):
        """Test when vector_processing config is not present."""
        # Mock Milvus connection utilities
        mock_connections.has_connection.return_value = True

        self.state["uploaded_files"] = []
        self.state["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
        self.state["selections"] = {}

        # Mock Collection for nodes and edges
        colls = {}
        colls["nodes"] = MagicMock()
        colls["nodes"].query.return_value = [
            {
                "node_index": 0,
                "node_id": "id1",
                "node_name": "JAK1",
                "node_type": "gene/protein",
                "feat": "featA",
                "feat_emb": [0.1, 0.2, 0.3],
                "desc": "descA",
                "desc_emb": [0.1, 0.2, 0.3],
            }
        ]
        colls["nodes"].load.return_value = None

        colls["edges"] = MagicMock()
        colls["edges"].query.return_value = [
            {
                "triplet_index": 0,
                "head_id": "id1",
                "tail_id": "id2",
                "edge_type": "gene/protein,ppi,gene/protein",
            }
        ]
        colls["edges"].load.return_value = None

        def collection_side_effect(name):
            if "nodes" in name:
                return colls["nodes"]
            if "edges" in name:
                return colls["edges"]
            return None

        mock_collection.side_effect = collection_side_effect

        # Mock MultimodalPCSTPruning
        mock_pcst_instance = MagicMock()
        mock_pcst_instance.extract_subgraph.return_value = {
            "nodes": pd.Series([1]),
            "edges": pd.Series([0]),
        }
        mock_pcst.return_value = mock_pcst_instance

        # Create config without vector_processing attribute
        cfg_no_vector_processing = MagicMock()
        cfg_no_vector_processing.cost_e = 1.0
        cfg_no_vector_processing.c_const = 1.0
        cfg_no_vector_processing.root = 0
        cfg_no_vector_processing.num_clusters = 1
        cfg_no_vector_processing.pruning = True
        cfg_no_vector_processing.verbosity_level = 0
        cfg_no_vector_processing.search_metric_type = "L2"
        cfg_no_vector_processing.node_colors_dict = {"gene/protein": "red"}
        # Remove vector_processing attribute to test the missing branch
        del cfg_no_vector_processing.vector_processing

        # Patch hydra.compose to return config without vector_processing
        with (
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.initialize"
            ),
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.compose"
            ) as mock_compose,
        ):
            mock_compose.return_value = MagicMock()
            mock_compose.return_value.app.frontend = self.cfg_db
            mock_compose.return_value.tools.multimodal_subgraph_extraction = (
                cfg_no_vector_processing
            )

            response = self.tool.invoke(
                input={
                    "prompt": self.prompt,
                    "tool_call_id": "subgraph_extraction_tool",
                    "state": self.state,
                    "arg_data": self.arg_data,
                }
            )

        # Verify the test completed successfully
        self.assertEqual(response.update["messages"][-1].tool_call_id, "subgraph_extraction_tool")

        # Test the collection_side_effect with unknown name for final test
        result = collection_side_effect("final_unknown_collection")
        self.assertIsNone(result)

        # Test the collection_side_effect with unknown name
        result = collection_side_effect("unknown_collection")
        self.assertIsNone(result)

    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.Collection"
    )
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.tools."
        "milvus_multimodal_subgraph_extraction.MultimodalPCSTPruning"
    )
    @patch("pymilvus.connections")
    def test_extract_multimodal_subgraph_dynamic_metrics_disabled(
        self, mock_connections, mock_pcst, mock_collection
    ):
        """Test when dynamic_metrics is disabled."""
        # Mock Milvus connection utilities
        mock_connections.has_connection.return_value = True

        self.state["uploaded_files"] = []
        self.state["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
        self.state["selections"] = {}

        # Mock Collection for nodes and edges
        colls = {}
        colls["nodes"] = MagicMock()
        colls["nodes"].query.return_value = [
            {
                "node_index": 0,
                "node_id": "id1",
                "node_name": "JAK1",
                "node_type": "gene/protein",
                "feat": "featA",
                "feat_emb": [0.1, 0.2, 0.3],
                "desc": "descA",
                "desc_emb": [0.1, 0.2, 0.3],
            }
        ]
        colls["nodes"].load.return_value = None

        colls["edges"] = MagicMock()
        colls["edges"].query.return_value = [
            {
                "triplet_index": 0,
                "head_id": "id1",
                "tail_id": "id2",
                "edge_type": "gene/protein,ppi,gene/protein",
            }
        ]
        colls["edges"].load.return_value = None

        def collection_side_effect(name):
            if "nodes" in name:
                return colls["nodes"]
            if "edges" in name:
                return colls["edges"]
            return None

        mock_collection.side_effect = collection_side_effect

        # Mock MultimodalPCSTPruning
        mock_pcst_instance = MagicMock()
        mock_pcst_instance.extract_subgraph.return_value = {
            "nodes": pd.Series([1]),
            "edges": pd.Series([0]),
        }
        mock_pcst.return_value = mock_pcst_instance

        # Create config with dynamic_metrics disabled
        cfg_dynamic_disabled = MagicMock()
        cfg_dynamic_disabled.cost_e = 1.0
        cfg_dynamic_disabled.c_const = 1.0
        cfg_dynamic_disabled.root = 0
        cfg_dynamic_disabled.num_clusters = 1
        cfg_dynamic_disabled.pruning = True
        cfg_dynamic_disabled.verbosity_level = 0
        cfg_dynamic_disabled.search_metric_type = "L2"
        cfg_dynamic_disabled.node_colors_dict = {"gene/protein": "red"}
        # Set dynamic_metrics to False
        cfg_dynamic_disabled.vector_processing = MagicMock()
        cfg_dynamic_disabled.vector_processing.dynamic_metrics = False

        # Patch hydra.compose to return config with dynamic_metrics disabled
        with (
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.initialize"
            ),
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.compose"
            ) as mock_compose,
        ):
            mock_compose.return_value = MagicMock()
            mock_compose.return_value.app.frontend = self.cfg_db
            mock_compose.return_value.tools.multimodal_subgraph_extraction = cfg_dynamic_disabled

            response = self.tool.invoke(
                input={
                    "prompt": self.prompt,
                    "tool_call_id": "subgraph_extraction_tool",
                    "state": self.state,
                    "arg_data": self.arg_data,
                }
            )

        # Verify the test completed successfully
        self.assertEqual(response.update["messages"][-1].tool_call_id, "subgraph_extraction_tool")

        # Test the collection_side_effect with unknown name for final test
        result = collection_side_effect("final_unknown_collection")
        self.assertIsNone(result)
