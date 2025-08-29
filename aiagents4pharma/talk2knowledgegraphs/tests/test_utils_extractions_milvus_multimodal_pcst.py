"""
Test cases for tools/utils/extractions/milvus_multimodal_pcst.py
"""

import importlib
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from ..utils.extractions.milvus_multimodal_pcst import (
    DynamicLibraryLoader,
    MultimodalPCSTPruning,
    SystemDetector,
)


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
        pcst_fast_patcher = patch(
            "aiagents4pharma.talk2knowledgegraphs.utils."
            "extractions.milvus_multimodal_pcst.pcst_fast"
        )
        mock_pcst_fast = pcst_fast_patcher.start()
        self.addCleanup(pcst_fast_patcher.stop)
        mock_pcst_fast.pcst_fast.return_value = ([0, 1], [0])

        # Patch Collection
        collection_patcher = patch(
            "aiagents4pharma.talk2knowledgegraphs.utils."
            "extractions.milvus_multimodal_pcst.Collection"
        )
        self.mock_collection = collection_patcher.start()
        self.addCleanup(collection_patcher.stop)

        # Patch open for cache_edge_index_path
        open_patcher = patch("builtins.open", mock_open(read_data="[[0,1],[1,2]]"))
        open_patcher.start()
        self.addCleanup(open_patcher.stop)

        # Patch pickle.load to return a numpy array for edge_index
        pickle_patcher = patch(
            "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.pickle"
        )
        mock_pickle = pickle_patcher.start()
        self.addCleanup(pickle_patcher.stop)
        mock_pickle.load.return_value = np.array([[0, 1], [1, 2]])

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

        # Setup mock loader
        self.mock_loader = MagicMock()
        self.mock_loader.py = np  # Use numpy for array operations
        self.mock_loader.df = pd  # Use pandas for dataframes
        self.mock_loader.to_list = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)

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
            loader=self.mock_loader,
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
            loader=self.mock_loader,
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
            loader=self.mock_loader,
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
            # Create local mocks for this test
            mock_pcst_fast = MagicMock()
            mock_pcst_fast.pcst_fast.return_value = ([0, 1], [0])
            mock_pickle = MagicMock()
            mock_pickle.load.return_value = np.array([[0, 1], [1, 2]])
            # Patch Collection, pcst_fast, and pickle after reload
            with (
                patch(f"{module_name}.Collection", self.mock_collection),
                patch(f"{module_name}.pcst_fast", mock_pcst_fast),
                patch(f"{module_name}.pickle", mock_pickle),
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
                    loader=self.mock_loader,
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


class TestSystemDetector(unittest.TestCase):
    """Test cases for SystemDetector class."""

    @patch("aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.platform")
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.subprocess"
    )
    def test_system_detector_gpu_detected(self, mock_subprocess, mock_platform):
        """Test SystemDetector when GPU is detected."""
        # Mock platform calls
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"

        # Mock successful nvidia-smi call
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.run.return_value = mock_result

        detector = SystemDetector()

        # Assertions
        self.assertEqual(detector.os_type, "linux")
        self.assertEqual(detector.architecture, "x86_64")
        self.assertTrue(detector.has_nvidia_gpu)
        self.assertTrue(detector.use_gpu)

    @patch("aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.platform")
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.subprocess"
    )
    def test_system_detector_no_gpu(self, mock_subprocess, mock_platform):
        """Test SystemDetector when no GPU is detected."""
        # Mock platform calls
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"

        # Mock failed nvidia-smi call
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.run.return_value = mock_result

        detector = SystemDetector()

        # Assertions
        self.assertEqual(detector.os_type, "linux")
        self.assertEqual(detector.architecture, "x86_64")
        self.assertFalse(detector.has_nvidia_gpu)
        self.assertFalse(detector.use_gpu)

    @patch("aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.platform")
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.subprocess"
    )
    def test_system_detector_macos_no_gpu(self, mock_subprocess, mock_platform):
        """Test SystemDetector on macOS (no GPU support)."""
        # Mock platform calls
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"

        # Mock successful nvidia-smi call (but macOS should still disable GPU)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.run.return_value = mock_result

        detector = SystemDetector()

        # Assertions
        self.assertEqual(detector.os_type, "darwin")
        self.assertEqual(detector.architecture, "arm64")
        self.assertTrue(detector.has_nvidia_gpu)  # GPU detected
        self.assertFalse(detector.use_gpu)  # But not used on macOS

    @patch("aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.platform")
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.subprocess"
    )
    def test_system_detector_subprocess_exception(self, mock_subprocess, mock_platform):
        """Test SystemDetector when subprocess raises exception."""
        # Mock platform calls
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"

        # Mock subprocess to raise FileNotFoundError
        mock_subprocess.run.side_effect = FileNotFoundError("nvidia-smi not found")
        mock_subprocess.TimeoutExpired = Exception
        mock_subprocess.SubprocessError = Exception

        detector = SystemDetector()

        # Assertions
        self.assertEqual(detector.os_type, "linux")
        self.assertEqual(detector.architecture, "x86_64")
        self.assertFalse(detector.has_nvidia_gpu)
        self.assertFalse(detector.use_gpu)

    @patch("aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.platform")
    @patch(
        "aiagents4pharma.talk2knowledgegraphs.utils.extractions.milvus_multimodal_pcst.subprocess"
    )
    def test_system_detector_timeout(self, mock_subprocess, mock_platform):
        """Test SystemDetector when subprocess times out."""
        # Mock platform calls
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"

        # Mock subprocess to raise TimeoutExpired
        mock_subprocess.TimeoutExpired = Exception
        mock_subprocess.SubprocessError = Exception
        mock_subprocess.run.side_effect = mock_subprocess.TimeoutExpired("nvidia-smi", 10)

        detector = SystemDetector()

        # Assertions
        self.assertEqual(detector.os_type, "linux")
        self.assertEqual(detector.architecture, "x86_64")
        self.assertFalse(detector.has_nvidia_gpu)
        self.assertFalse(detector.use_gpu)

    def test_get_system_info(self):
        """Test get_system_info method."""
        with (
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
                "milvus_multimodal_pcst.platform"
            ) as mock_platform,
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
                "milvus_multimodal_pcst.subprocess"
            ) as mock_subprocess,
        ):
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_subprocess.run.return_value = mock_result

            detector = SystemDetector()
            info = detector.get_system_info()

            expected_info = {
                "os_type": "linux",
                "architecture": "x86_64",
                "has_nvidia_gpu": True,
                "use_gpu": True,
            }
            self.assertEqual(info, expected_info)

    def test_is_gpu_compatible(self):
        """Test is_gpu_compatible method."""
        with (
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
                "milvus_multimodal_pcst.platform"
            ) as mock_platform,
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
                "milvus_multimodal_pcst.subprocess"
            ) as mock_subprocess,
        ):
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_subprocess.run.return_value = mock_result

            detector = SystemDetector()

            # Should be compatible (has GPU + not macOS)
            self.assertTrue(detector.is_gpu_compatible())


class TestDynamicLibraryLoader(unittest.TestCase):
    """Test cases for DynamicLibraryLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_detector = MagicMock()

    def test_dynamic_library_loader_cpu_mode(self):
        """Test DynamicLibraryLoader in CPU mode."""
        self.mock_detector.use_gpu = False

        loader = DynamicLibraryLoader(self.mock_detector)

        # Assertions
        self.assertFalse(loader.use_gpu)
        self.assertEqual(loader.py, np)
        self.assertEqual(loader.df, pd)
        self.assertFalse(loader.normalize_vectors)
        self.assertEqual(loader.metric_type, "COSINE")

    @patch.dict("sys.modules", {"cupy": MagicMock(), "cudf": MagicMock()})
    def test_dynamic_library_loader_gpu_mode_success(self):
        """Test DynamicLibraryLoader in GPU mode with successful imports."""
        self.mock_detector.use_gpu = True

        # Mock the CUDF_AVAILABLE flag
        with patch(
            "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
            "milvus_multimodal_pcst.CUDF_AVAILABLE",
            True,
        ):
            loader = DynamicLibraryLoader(self.mock_detector)

        # Assertions
        self.assertTrue(loader.use_gpu)
        self.assertTrue(loader.normalize_vectors)
        self.assertEqual(loader.metric_type, "IP")

    def test_dynamic_library_loader_gpu_mode_import_failure(self):
        """Test DynamicLibraryLoader when GPU libraries are not available."""
        self.mock_detector.use_gpu = True

        # Mock CUDF_AVAILABLE as False to simulate import failure
        with patch(
            "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
            "milvus_multimodal_pcst.CUDF_AVAILABLE",
            False,
        ):
            loader = DynamicLibraryLoader(self.mock_detector)

        # Should fallback to CPU mode
        self.assertFalse(loader.use_gpu)
        self.assertEqual(loader.py, np)
        self.assertEqual(loader.df, pd)
        self.assertFalse(loader.normalize_vectors)
        self.assertEqual(loader.metric_type, "COSINE")

    def test_normalize_matrix_cpu_mode(self):
        """Test normalize_matrix in CPU mode."""
        self.mock_detector.use_gpu = False
        loader = DynamicLibraryLoader(self.mock_detector)

        matrix = np.array([[1, 2], [3, 4]])
        result = loader.normalize_matrix(matrix)

        # In CPU mode, should return matrix unchanged
        np.testing.assert_array_equal(result, matrix)

    def test_normalize_matrix_cpu_mode_normalize_false(self):
        """Test normalize_matrix in CPU mode with normalize_vectors=False explicitly."""
        self.mock_detector.use_gpu = False
        loader = DynamicLibraryLoader(self.mock_detector)

        # Explicitly set normalize_vectors to False to test the return path
        loader.normalize_vectors = False

        matrix = np.array([[1, 2], [3, 4]])
        result = loader.normalize_matrix(matrix)

        # Should return matrix unchanged when normalize_vectors is False
        np.testing.assert_array_equal(result, matrix)

    def test_normalize_matrix_cpu_mode_normalize_true(self):
        """Test normalize_matrix in CPU mode with normalize_vectors=True (edge case)."""
        self.mock_detector.use_gpu = False
        loader = DynamicLibraryLoader(self.mock_detector)

        # Force normalize_vectors to True to test the else branch at line 144
        loader.normalize_vectors = True
        loader.use_gpu = False  # Ensure GPU is disabled

        matrix = np.array([[1, 2], [3, 4]])
        result = loader.normalize_matrix(matrix)

        # Should return matrix unchanged in CPU mode even when normalize_vectors is True
        np.testing.assert_array_equal(result, matrix)

    @patch.dict("sys.modules", {"cupy": MagicMock(), "cudf": MagicMock()})
    def test_normalize_matrix_gpu_mode(self):
        """Test normalize_matrix in GPU mode."""
        self.mock_detector.use_gpu = True

        with patch(
            "aiagents4pharma.talk2knowledgegraphs.utils.extractions."
            "milvus_multimodal_pcst.CUDF_AVAILABLE",
            True,
        ):
            loader = DynamicLibraryLoader(self.mock_detector)

            # Mock cupy operations
            mock_cp = MagicMock()
            mock_array = MagicMock()
            mock_norms = MagicMock()

            mock_cp.asarray.return_value = mock_array
            mock_cp.linalg.norm.return_value = mock_norms
            mock_cp.float32 = np.float32

            loader.cp = mock_cp
            loader.py = mock_cp

            matrix = [[1, 2], [3, 4]]
            loader.normalize_matrix(matrix)

            # Verify cupy operations were called
            mock_cp.asarray.assert_called_once()
            mock_cp.linalg.norm.assert_called_once()

    def test_to_list_with_tolist(self):
        """Test to_list with data that has tolist method."""
        self.mock_detector.use_gpu = False
        loader = DynamicLibraryLoader(self.mock_detector)

        data = np.array([1, 2, 3])
        result = loader.to_list(data)

        self.assertEqual(result, [1, 2, 3])

    def test_to_list_with_to_arrow(self):
        """Test to_list with data that has to_arrow method."""
        self.mock_detector.use_gpu = False
        loader = DynamicLibraryLoader(self.mock_detector)

        # Mock data with to_arrow method but no tolist method
        mock_data = MagicMock()
        mock_arrow = MagicMock()
        mock_arrow.to_pylist.return_value = [1, 2, 3]
        mock_data.to_arrow.return_value = mock_arrow
        # Remove tolist method to test the to_arrow path
        del mock_data.tolist

        result = loader.to_list(mock_data)

        self.assertEqual(result, [1, 2, 3])

    def test_to_list_fallback(self):
        """Test to_list fallback to list()."""
        self.mock_detector.use_gpu = False
        loader = DynamicLibraryLoader(self.mock_detector)

        data = (1, 2, 3)  # tuple without tolist or to_arrow
        result = loader.to_list(data)

        self.assertEqual(result, [1, 2, 3])
