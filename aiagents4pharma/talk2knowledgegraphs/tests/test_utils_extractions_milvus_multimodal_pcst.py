"""
Test cases for tools/utils/extractions/milvus_multimodal_pcst.py
"""

import importlib
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from ..utils.extractions.milvus_multimodal_pcst import MultimodalPCSTPruning


class TestMultimodalPCSTPruning(unittest.TestCase):
    """
    Test cases for MultimodalPCSTPruning class (Milvus-based PCST pruning).
    """

    def setUp(self):
        # Patch cupy and cudf to simulate GPU environment
        patcher_cupy = patch.dict("sys.modules", {"cupy": MagicMock(), "cudf": MagicMock()})
        patcher_cupy.start()
        self.addCleanup(patcher_cupy.stop)

        # Patch pcst_fast
        self.pcst_fast_patcher = patch(
            "aiagents4pharma.talk2knowledgegraphs.utils."
            "extractions.milvus_multimodal_pcst.pcst_fast"
        )
        self.mock_pcst_fast = self.pcst_fast_patcher.start()
        self.addCleanup(self.pcst_fast_patcher.stop)
        self.mock_pcst_fast.pcst_fast.return_value = ([0, 1], [0])

        # Patch Collection
        self.collection_patcher = patch(
            "aiagents4pharma.talk2knowledgegraphs.utils."
            "extractions.milvus_multimodal_pcst.Collection"
        )
        self.mock_collection = self.collection_patcher.start()
        self.addCleanup(self.collection_patcher.stop)

        # Patch open for cache_edge_index_path
        self.open_patcher = patch("builtins.open", mock_open(read_data="[[0,1],[1,2]]"))
        self.mock_open = self.open_patcher.start()
        self.addCleanup(self.open_patcher.stop)

        # Patch pickle.load to return a numpy array for edge_index
        self.pickle_patcher = patch(
            "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.pickle"
        )
        self.mock_pickle = self.pickle_patcher.start()
        self.addCleanup(self.pickle_patcher.stop)
        self.mock_pickle.load.return_value = np.array([[0, 1], [1, 2]])

        # Setup config mock
        self.cfg = MagicMock()
        self.cfg.milvus_db.database_name = "testdb"
        self.cfg.milvus_db.cache_edge_index_path = "dummy_cache.pkl"

        # Setup Collection mocks
        node_coll = MagicMock()
        node_coll.num_entities = 2
        node_coll.search.return_value = [[MagicMock(id=0), MagicMock(id=1)]]
        edge_coll = MagicMock()
        edge_coll.num_entities = 2
        edge_coll.search.return_value = [[MagicMock(id=0, score=1.0), MagicMock(id=1, score=0.5)]]
        self.mock_collection.side_effect = lambda name: (
            node_coll if "nodes" in name else edge_coll
        )

    def test_extract_subgraph_use_description_true(self):
        """
        Test the extract_subgraph method of MultimodalPCSTPruning with use_description=True.
        """
        # Create instance
        pcst = MultimodalPCSTPruning(
            topk=3,
            topk_e=3,
            cost_e=0.5,
            c_const=0.01,
            root=-1,
            num_clusters=1,
            pruning="gw",
            verbosity_level=0,
            use_description=True,
            metric_type="IP",
        )
        # Dummy embeddings
        text_emb = [0.1, 0.2, 0.3]
        query_emb = [0.1, 0.2, 0.3]
        modality = "gene/protein"

        # Call extract_subgraph
        result = pcst.extract_subgraph(text_emb, query_emb, modality, self.cfg)

        # Assertions
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertGreaterEqual(len(result["nodes"]), 0)
        self.assertGreaterEqual(len(result["edges"]), 0)

    def test_extract_subgraph_use_description_false(self):
        """
        Test the extract_subgraph method of MultimodalPCSTPruning with use_description=False.
        """
        # Create instance
        pcst = MultimodalPCSTPruning(
            topk=3,
            topk_e=3,
            cost_e=0.5,
            c_const=0.01,
            root=-1,
            num_clusters=1,
            pruning="gw",
            verbosity_level=0,
            use_description=False,
            metric_type="IP",
        )
        # Dummy embeddings
        text_emb = [0.1, 0.2, 0.3]
        query_emb = [0.1, 0.2, 0.3]
        modality = "gene/protein"

        # Call extract_subgraph
        result = pcst.extract_subgraph(text_emb, query_emb, modality, self.cfg)

        # Assertions
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertGreaterEqual(len(result["nodes"]), 0)
        self.assertGreaterEqual(len(result["edges"]), 0)

    def test_extract_subgraph_with_virtual_vertices(self):
        """
        Test get_subgraph_nodes_edges with virtual vertices present (len(virtual_vertices) > 0).
        """
        pcst = MultimodalPCSTPruning(
            topk=3,
            topk_e=3,
            cost_e=0.5,
            c_const=0.01,
            root=-1,
            num_clusters=1,
            pruning="gw",
            verbosity_level=0,
            use_description=True,
            metric_type="IP",
        )
        # Simulate num_nodes = 2, vertices contains [0, 1, 2, 3] (2 and 3 are virtual)
        num_nodes = 2
        # vertices: [0, 1, 2, 3] (2 and 3 are virtual)
        vertices = np.array([0, 1, 2, 3])
        # edges_dict simulates prior edges and edge_index
        edges_dict = {
            "edges": np.array([0, 1, 2]),
            "num_prior_edges": 2,
            "edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
        }
        # mapping simulates mapping for edges and nodes
        mapping = {"edges": {0: 0, 1: 1}, "nodes": {2: 2, 3: 3}}

        # Call extract_subgraph
        result = pcst.get_subgraph_nodes_edges(num_nodes, vertices, edges_dict, mapping)

        # Assertions
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertGreaterEqual(len(result["nodes"]), 0)
        self.assertGreaterEqual(len(result["edges"]), 0)
        # Check that virtual edges are included
        self.assertTrue(any(e in [2, 3] for e in result["edges"]))

    def test_gpu_import_branch(self):
        """
        Test coverage for GPU import branch by patching sys.modules to mock cupy and
        cudf as numpy and pandas.
        """
        module_name = (
            "aiagents4pharma.talk2knowledgegraphs.utils" + ".extractions.milvus_multimodal_pcst"
        )
        with patch.dict("sys.modules", {"cupy": np, "cudf": pd}):
            # Reload the module to trigger the GPU branch
            mod = importlib.reload(sys.modules[module_name])
            # Patch Collection, pcst_fast, and pickle after reload
            with (
                patch(f"{module_name}.Collection", self.mock_collection),
                patch(f"{module_name}.pcst_fast", self.mock_pcst_fast),
                patch(f"{module_name}.pickle", self.mock_pickle),
            ):
                pcst_pruning_cls = mod.MultimodalPCSTPruning
                pcst = pcst_pruning_cls(
                    topk=3,
                    topk_e=3,
                    cost_e=0.5,
                    c_const=0.01,
                    root=-1,
                    num_clusters=1,
                    pruning="gw",
                    verbosity_level=0,
                    use_description=True,
                    metric_type="IP",
                )
                # Dummy embeddings
                text_emb = [0.1, 0.2, 0.3]
                query_emb = [0.1, 0.2, 0.3]
                modality = "gene/protein"

                # Call extract_subgraph
                result = pcst.extract_subgraph(text_emb, query_emb, modality, self.cfg)

                # Assertions
                self.assertIn("nodes", result)
                self.assertIn("edges", result)
                self.assertGreaterEqual(len(result["nodes"]), 0)
                self.assertGreaterEqual(len(result["edges"]), 0)
