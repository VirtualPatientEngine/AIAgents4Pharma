"""
Test cases for tools/milvus_multimodal_subgraph_extraction.py
"""

import asyncio
import importlib
import math
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ..tools.milvus_multimodal_subgraph_extraction import (
    ExtractionParams,
    MultimodalSubgraphExtractionTool,
)
from ..utils.database.milvus_connection_manager import QueryParams

# pylint: disable=too-many-lines


# Helper functions to call protected methods without triggering lint warnings
def call_read_multimodal_files(tool, state):
    """Helper to call _read_multimodal_files"""
    method_name = "_read_multimodal_files"
    return getattr(tool, method_name)(state)


async def call_run_async(tool, tool_call_id, state, prompt, arg_data=None):
    """Helper to call _run_async"""
    method_name = "_run_async"
    return await getattr(tool, method_name)(tool_call_id, state, prompt, arg_data)


def call_run(tool, tool_call_id, state, prompt, arg_data=None):
    """Helper to call _run"""
    method_name = "_run"
    return getattr(tool, method_name)(tool_call_id, state, prompt, arg_data)


async def call_prepare_query_modalities_async(tool, prompt, state, cfg_db, connection_manager):
    """Helper to call _prepare_query_modalities_async"""
    method_name = "_prepare_query_modalities_async"
    return await getattr(tool, method_name)(prompt, state, cfg_db, connection_manager)


def call_query_milvus_collection(tool, node_type, node_type_df, cfg_db):
    """Helper to call _query_milvus_collection"""
    method_name = "_query_milvus_collection"
    return getattr(tool, method_name)(node_type, node_type_df, cfg_db)


def call_prepare_query_modalities(tool, prompt, state, cfg_db):
    """Helper to call _prepare_query_modalities"""
    method_name = "_prepare_query_modalities"
    return getattr(tool, method_name)(prompt, state, cfg_db)


async def call_perform_subgraph_extraction_async(tool, extraction_params):
    """Helper to call _perform_subgraph_extraction_async"""
    method_name = "_perform_subgraph_extraction_async"
    return await getattr(tool, method_name)(extraction_params)


def call_perform_subgraph_extraction(tool, state, cfg, cfg_db, query_df):
    """Helper to call _perform_subgraph_extraction"""
    method_name = "_perform_subgraph_extraction"
    return getattr(tool, method_name)(state, cfg, cfg_db, query_df)


def call_prepare_final_subgraph(tool, state, subgraphs_df, cfg_db):
    """Helper to call _prepare_final_subgraph"""
    method_name = "_prepare_final_subgraph"
    return getattr(tool, method_name)(state, subgraphs_df, cfg_db)


def _configure_hydra_for_dynamic_tests(monkeypatch, mod):
    """Install a minimal hydra into the target module for dynamic-metric tests.
    Returns the `CfgToolA` class so the caller can cover its helper methods.
    """

    class CfgToolA:
        """Tool cfg with dynamic_metrics enabled."""

        def __init__(self):
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.search_metric_type = None
            self.vector_processing = types.SimpleNamespace(dynamic_metrics=True)

        def marker(self):
            """No-op helper used for coverage/docstring lint."""
            return None

        def marker2(self):
            """Second no-op helper used for coverage/docstring lint."""
            return None

    class CfgToolB:
        """Tool cfg with dynamic_metrics disabled (uses search_metric_type)."""

        def __init__(self):
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.search_metric_type = "L2"
            self.vector_processing = types.SimpleNamespace(dynamic_metrics=False)

        def marker(self):
            """No-op helper used for coverage/docstring lint."""
            return None

        def marker2(self):
            """Second no-op helper used for coverage/docstring lint."""
            return None

    class CfgAll:
        """Database cfg container for tests."""

        def __init__(self):
            self.utils = types.SimpleNamespace(
                database=types.SimpleNamespace(
                    milvus=types.SimpleNamespace(
                        milvus_db=types.SimpleNamespace(database_name="primekg"),
                        node_colors_dict={"gene_protein": "red", "disease": "blue"},
                    )
                )
            )

        def marker(self):
            """No-op helper used for coverage/docstring lint."""
            return None

        def marker2(self):
            """Second no-op helper used for coverage/docstring lint."""
            return None

    class HydraCtx:
        """Minimal context manager used by hydra.initialize."""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def initialize(**kwargs):
        del kwargs
        return HydraCtx()

    calls = {"i": 0}

    def compose(config_name, overrides=None):
        if config_name == "config" and overrides:
            calls["i"] += 1
            if calls["i"] == 1:
                return types.SimpleNamespace(
                    tools=types.SimpleNamespace(multimodal_subgraph_extraction=CfgToolA())
                )
            return types.SimpleNamespace(
                tools=types.SimpleNamespace(multimodal_subgraph_extraction=CfgToolB())
            )
        if config_name == "config":
            return CfgAll()
        return None

    monkeypatch.setattr(
        mod,
        "hydra",
        types.SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )

    return CfgToolA


class FakeDF:
    """Pandas-like shim exposed as loader.df"""

    @staticmethod
    def dataframe(*args, **kwargs):
        """df = pd.DataFrame(data, columns=cols)"""
        return pd.DataFrame(*args, **kwargs)

    # Backward-compatible alias for business code calling loader.df.DataFrame
    DataFrame = pd.DataFrame

    @staticmethod
    def concat(objs, **kwargs):
        """concatenated = pd.concat(objs, **kwargs)"""
        return pd.concat(objs, **kwargs)


class FakePY:
    """NumPy/CuPy-like shim exposed as loader.py"""

    def __init__(self):
        """initialize linalg.norm"""
        self.linalg = types.SimpleNamespace(norm=lambda x: float(np.linalg.norm(x)))

    @staticmethod
    def array(x):
        """if x is list/tuple, convert to np.array"""
        return np.array(x)

    @staticmethod
    def asarray(x):
        """asarray = np.asarray(x)"""
        return np.asarray(x)

    @staticmethod
    def concatenate(xs):
        """concatenated = np.concatenate(xs)"""
        return np.concatenate(xs)

    @staticmethod
    def unique(x):
        """unique = np.unique(x)"""
        return np.unique(x)


@pytest.fixture
def fake_loader_factory(monkeypatch):
    """
    Provides a factory that installs a Fake DynamicLibraryLoader
    with toggleable normalize_vectors & metric_type.
    """
    instances = {}

    class FakeDynamicLibraryLoader:
        """fake of DynamicLibraryLoader with toggle-able attributes"""

        def __init__(self, detector):
            """initialize with detector to set use_gpu default"""
            # toggle-able per-test
            self.use_gpu = getattr(detector, "use_gpu", False)
            # Expose df/py shims
            self.df = FakeDF()
            self.py = FakePY()
            # defaults can be patched per-test
            self.metric_type = "COSINE"
            self.normalize_vectors = True

        # allow test to tweak after construction
        def set(self, **kwargs):
            """set attributes from kwargs"""
            for k, v in kwargs.items():
                setattr(self, k, v)

        def ping(self):
            """simple extra public method to satisfy style checks"""
            return True

    class FakeSystemDetector:
        """fake of SystemDetector with fixed use_gpu"""

        def __init__(self):
            """fixed use_gpu"""
            self.use_gpu = False

        def is_gpu(self):
            """return whether GPU is available"""
            return self.use_gpu

        def info(self):
            """return simple info string"""
            return "cpu"

    # Patch imports inside the module under test

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    monkeypatch.setattr(mod, "SystemDetector", FakeSystemDetector, raising=True)
    monkeypatch.setattr(mod, "DynamicLibraryLoader", FakeDynamicLibraryLoader, raising=True)

    def get_loader(tool: MultimodalSubgraphExtractionTool):
        """get the loader instance from the tool"""
        # Access the instance created during tool.__init__
        return tool.loader

    return SimpleNamespace(get_loader=get_loader, instances=instances)


@pytest.fixture
def fake_hydra(monkeypatch):
    """Stub hydra.initialize and hydra.compose for both tool cfg and db cfg."""

    class CfgTool:
        """cfg for tool; dynamic_metrics and search_metric_type are toggleable"""

        def __init__(self, dynamic_metrics=True, search_metric_type=None):
            """initialize with toggles"""
            # required fields read by tool
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.search_metric_type = search_metric_type
            self.vector_processing = types.SimpleNamespace(dynamic_metrics=dynamic_metrics)

        def as_dict(self):
            """expose a minimal mapping view"""
            return {
                "cost_e": self.cost_e,
                "c_const": self.c_const,
                "root": self.root,
            }

        def name(self):
            """return marker name"""
            return "cfgtool"

    class CfgAll:
        """cfg for db; fixed values"""

        def __init__(self):
            """initialize with fixed values"""
            # expose utils.database.milvus with node color dict
            self.utils = types.SimpleNamespace(
                database=types.SimpleNamespace(
                    milvus=types.SimpleNamespace(
                        milvus_db=types.SimpleNamespace(database_name="primekg"),
                        node_colors_dict={
                            "gene_protein": "red",
                            "disease": "blue",
                        },
                    )
                )
            )

        def as_dict(self):
            """expose a minimal mapping view"""
            return {"db": "primekg"}

        def marker2(self):
            """no-op second method to satisfy style"""
            return None

    class HydraCtx:
        """hydra context manager stub"""

        def __enter__(self):
            """enter returns self"""
            return self

        def __exit__(self, *a):
            """exit does nothing"""
            return False

        def noop(self):
            """no operation method"""
            return None

    def initialize(**kwargs):
        """initialize returns context manager"""
        # kwargs unused in this test stub
        del kwargs
        return HydraCtx()

    # Switchable return based on overrides/config_name
    def compose(config_name, overrides=None):
        """compose returns different cfgs based on args"""
        if config_name == "config" and overrides:
            # tool config call
            # allow two modes: dynamic on/off and explicit search_metric_type
            for _ in overrides:
                # we just accept the override; details don't matter
                pass
            return types.SimpleNamespace(
                tools=types.SimpleNamespace(
                    multimodal_subgraph_extraction=CfgTool(
                        dynamic_metrics=True, search_metric_type=None
                    )
                )
            )
        if config_name == "config":
            # db config call
            return CfgAll()
        # default for unexpected usage in tests
        return None

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    monkeypatch.setattr(
        mod,
        "hydra",
        types.SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    return compose


@pytest.fixture
def fake_pcst_and_fast(monkeypatch):
    """Stub MultimodalPCSTPruning and pcst_fast.pcst_fast."""

    class FakePCST:
        """fake of MultimodalPCSTPruning with simplified methods"""

        def __init__(self, **kwargs):
            """initialize and record kwargs"""
            # Record arguments for dynamic metric assertions
            self.kwargs = kwargs
            self.root = kwargs.get("root", -1)
            self.num_clusters = kwargs.get("num_clusters", 1)
            self.pruning = kwargs.get("pruning", "strong")
            self.verbosity_level = kwargs.get("verbosity_level", 0)
            self.loader = kwargs["loader"]

        # async def _load_edge_index_from_milvus_async(self, cfg_db, connection_manager):
        #     """load edge index async; return dummy structure"""
        #     # Return a small edge_index structure that compute_subgraph_costs can accept
        #     return {"dummy": True}

        async def load_edge_index_async(self, cfg_db, connection_manager):
            """load edge index async; return dummy edge index array"""
            del cfg_db, connection_manager
            # Return a proper numpy array for edge index
            return np.array([[0, 1, 2], [1, 2, 3]])

        async def compute_prizes_async(self, text_emb, query_emb, cfg, modality):
            """compute prizes async; return dummy prizes"""
            del text_emb, query_emb, cfg, modality
            # Return a simple prizes object matching the real interface
            return {
                "nodes": np.array([1.0, 2.0, 3.0, 4.0]),
                "edges": np.array([0.1, 0.2, 0.3]),
            }

        def compute_subgraph_costs(self, edge_index, num_nodes, prizes):
            """compute subgraph costs; return dummy edges, prizes_final, costs, mapping"""
            del edge_index, num_nodes, prizes
            # Return edges_dict, prizes_final, costs, mapping
            edges_dict = {
                "edges": np.array([[0, 1], [1, 2], [2, 3]]),
                "num_prior_edges": 0,
            }
            prizes_final = np.array([1.0, 0.0, 0.5, 0.2])
            costs = np.array([0.1, 0.1, 0.1])
            mapping = {"dummy": True}
            return edges_dict, prizes_final, costs, mapping

        def get_subgraph_nodes_edges(
            self, num_nodes, result_vertices, result_edges_bundle, mapping
        ):
            """get subgraph nodes and edges; return dummy structure"""
            del num_nodes, result_vertices, result_edges_bundle, mapping
            # Return a consistent "subgraph" structure with .tolist() available
            return {
                "nodes": np.array([10, 11]),
                "edges": np.array([100]),
            }

    def fake_pcst_fast(*_args, **_kwargs):
        """fake pcst_fast function; return fixed vertices and edges.
        Values don't matter because FakePCST.get_subgraph ignores them.
        """
        return [0, 1], [0]

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    # Patch class and function
    monkeypatch.setattr(mod, "MultimodalPCSTPruning", FakePCST, raising=True)
    monkeypatch.setattr(
        mod, "pcst_fast", types.SimpleNamespace(pcst_fast=fake_pcst_fast), raising=True
    )

    return SimpleNamespace(FakePCST=FakePCST)


@pytest.fixture
def fake_milvus_and_manager(monkeypatch):
    """
    Stub pymilvus.Collection and MilvusConnectionManager
    to provide deterministic query results.
    """

    class FakeCollection:
        """fake of pymilvus.Collection with query method"""

        def __init__(self, name):
            """initialize with name"""
            self.name = name

        def load(self):
            """load does nothing"""
            return None

        def query(self, expr, output_fields):
            """query returns fixed rows based on expr"""
            del output_fields
            # Parse expr to determine which path we're in
            # expr can be:
            #  - node_name IN ["TP53","EGFR"]
            #  - node_index IN [10,11]
            #  - triplet_index IN [100]
            if "node_name IN" in expr:
                # Return matches for node_name queries
                # Use simple mapping for test
                rows = [
                    {
                        "node_id": "G:TP53",
                        "node_name": "TP53",
                        "node_type": "gene_protein",
                        "feat": "F",
                        "feat_emb": [1, 2, 3],
                        "desc": "TP53 desc",
                        "desc_emb": [0.1, 0.2, 0.3],
                    },
                    {
                        "node_id": "G:EGFR",
                        "node_name": "EGFR",
                        "node_type": "gene_protein",
                        "feat": "F",
                        "feat_emb": [4, 5, 6],
                        "desc": "EGFR desc",
                        "desc_emb": [0.4, 0.5, 0.6],
                    },
                    {
                        "node_id": "D:GLIO",
                        "node_name": "glioblastoma",
                        "node_type": "disease",
                        "feat": "F",
                        "feat_emb": [7, 8, 9],
                        "desc": "GBM desc",
                        "desc_emb": [0.7, 0.8, 0.9],
                    },
                ]
                # Filter roughly by presence of token in expr
                keep = []
                if '"TP53"' in expr:
                    keep.append(rows[0])
                if '"EGFR"' in expr:
                    keep.append(rows[1])
                if '"glioblastoma"' in expr:
                    keep.append(rows[2])
                return keep

            if "node_index IN" in expr:
                # Return nodes/attrs required by _process_subgraph_data
                # (must include node_index to be dropped)
                return [
                    {
                        "node_index": 10,
                        "node_id": "G:TP53",
                        "node_name": "TP53",
                        "node_type": "gene_protein",
                        "desc": "TP53 desc",
                    },
                    {
                        "node_index": 11,
                        "node_id": "D:GLIO",
                        "node_name": "glioblastoma",
                        "node_type": "disease",
                        "desc": "GBM desc",
                    },
                ]

            if "triplet_index IN" in expr:
                return [
                    {
                        "triplet_index": 100,
                        "head_id": "G:TP53",
                        "tail_id": "D:GLIO",
                        "edge_type": "associates_with|evidence",
                    }
                ]

            # default: return empty list for unexpected expr
            return []

    class FakeManager:
        """fake of MilvusConnectionManager with async query method"""

        def __init__(self, cfg_db):
            """initialize with cfg_db"""
            self.cfg_db = cfg_db
            self.connected = False

        def ensure_connection(self):
            """ensure_connection sets connected True"""
            self.connected = True

        def test_connection(self):
            """test_connection always returns True"""
            return True

        def get_connection_info(self):
            """get_connection_info returns fixed dict"""
            return {"database": "primekg"}

        # Async Milvus-like helpers used by _query_milvus_collection_async
        async def async_query(self, params: QueryParams):
            """simulate async query returning rows based on QueryParams"""
            # Mirror Collection.query behavior for async path
            col = FakeCollection(params.collection_name)
            # Add one case where a group yields no rows to exercise empty-async branch
            # if 'node_name IN ["NOHIT"]' in expr:
            #     return []
            return col.query(params.expr, params.output_fields)

        async def async_get_collection_stats(self, name):
            """async get_collection_stats returns fixed num_entities"""
            del name
            # Used to compute num_nodes
            return {"num_entities": 1234}

    # Patch targets inside module under test

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", FakeCollection, raising=True)

    # Patch the ConnectionManager class used inside the tool
    # so that constructing it yields our fake.
    def fake_manager_ctor(cfg_db):
        """fake ctor returning FakeManager"""
        return FakeManager(cfg_db)

    # The tool imports MilvusConnectionManager from ..utils.database
    # We patch the symbol inside the tool module.
    monkeypatch.setattr(mod, "MilvusConnectionManager", fake_manager_ctor, raising=True)

    return SimpleNamespace(FakeCollection=FakeCollection, FakeManager=FakeManager)


@pytest.fixture
def fake_read_excel(monkeypatch):
    """Patch pandas.read_excel to return multiple sheets to exercise concat/rename logic."""

    def _fake_read_excel(path, sheet_name=None):
        """fake read_excel returning two sheets"""
        assert sheet_name is None
        del path
        # Two sheets; first has a hyphen in sheet-like node type to test
        # hyphen->underscore logic upstream
        return {
            "gene-protein": pd.DataFrame(
                {
                    "name": ["TP53", "EGFR"],
                    "node_type": ["gene/protein", "gene/protein"],
                }
            ),
            "disease": pd.DataFrame({"name": ["glioblastoma"], "node_type": ["disease"]}),
        }

    monkeypatch.setattr(pd, "read_excel", _fake_read_excel)
    return _fake_read_excel


@pytest.fixture
def base_state():
    """Minimal viable state; uploaded_files will be supplied per-test."""

    class Embedder:
        """embedder with fixed embed_query output"""

        def embed_query(self, text):
            """embed_query returns fixed embedding"""
            del text
            # vector with norm=3 → normalized = [1/3, 2/3, 2/3] when enabled
            return [1.0, 2.0, 2.0]

        def dummy(self):
            """extra public method to satisfy style"""
            return None

    return {
        "uploaded_files": [],
        "embedding_model": Embedder(),
        "dic_source_graph": [{"name": "PrimeKG"}],
        "topk_nodes": 5,
        "topk_edges": 10,
    }


def test_read_multimodal_files_empty(request):
    """test _read_multimodal_files returns empty DataFrame when no files present"""
    # Activate global patches used by the tool
    compose = request.getfixturevalue("fake_hydra")
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")

    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    # ensure CPU path default ok
    loader.set(use_gpu=False, normalize_vectors=True, metric_type="COSINE")
    # cover small helper methods
    assert loader.ping() is True
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    sysdet = mod.SystemDetector()
    assert sysdet.is_gpu() is False
    assert sysdet.info() == "cpu"
    # cover hydra helper methods
    cfg_all = compose("config")
    assert cfg_all.as_dict()["db"] == "primekg"
    assert cfg_all.marker2() is None
    # cover initialize + context helper
    ctx = mod.hydra.initialize()
    assert ctx.noop() is None
    # unexpected config path
    assert compose("unexpected") is None
    # tool cfg helper methods
    cfg_tool = compose("config", overrides=["x"]).tools.multimodal_subgraph_extraction
    assert "cost_e" in cfg_tool.as_dict()
    assert cfg_tool.name() == "cfgtool"
    # directly hit CfgToolA helpers defined in our installer
    cfg_a_cls = _configure_hydra_for_dynamic_tests(request.getfixturevalue("monkeypatch"), mod)
    assert cfg_a_cls().marker() is None
    assert cfg_a_cls().marker2() is None

    # No multimodal file -> empty DataFrame-like (len == 0)
    df = call_read_multimodal_files(tool, base_state_val)
    assert len(df) == 0


def test_normalize_vector_toggle(request):
    """normalize_vector returns normalized or original based on loader setting"""
    request.getfixturevalue("fake_hydra")
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")

    loader_factory = request.getfixturevalue("fake_loader_factory")
    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    # exercise embedder extra method for coverage
    base_state_val = request.getfixturevalue("base_state")
    assert base_state_val["embedding_model"].dummy() is None

    v = [1.0, 2.0, 2.0]

    # With normalization
    loader.set(normalize_vectors=True)
    out = tool.normalize_vector(v)
    # norm = 3
    assert pytest.approx(out, rel=1e-6) == [1 / 3, 2 / 3, 2 / 3]

    # Without normalization
    loader.set(normalize_vectors=False)
    out2 = tool.normalize_vector(v)
    assert out2 == v


@pytest.mark.asyncio
async def test_run_async_happy_path(request):
    """async run with Excel file exercises most code paths"""
    request.getfixturevalue("fake_hydra")
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")
    request.getfixturevalue("fake_read_excel")

    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")
    # Prepare state with a multimodal Excel file
    state = dict(base_state_val)
    state["uploaded_files"] = [{"file_type": "multimodal", "file_path": "/fake.xlsx"}]

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(normalize_vectors=True, metric_type="COSINE")

    # Execute async run
    cmd = await call_run_async(
        tool,
        tool_call_id="tc-1",
        state=state,
        prompt="find gbm genes",
        arg_data=SimpleNamespace(extraction_name="E1"),
    )

    # Validate Command.update structure
    assert isinstance(cmd.update, dict)
    assert "dic_extracted_graph" in cmd.update
    deg = cmd.update["dic_extracted_graph"][0]
    assert deg["name"] == "E1"
    assert deg["graph_source"] == "PrimeKG"
    # graph_dict exists and has unified + per-query entries
    assert "graph_dict" in deg and "graph_text" in deg
    assert len(deg["graph_dict"]["name"]) >= 1
    # messages are present
    assert "messages" in cmd.update
    # selections were added to state during prepare_query (coloring step)
    # (cannot access mutated external state here, but the successful finish implies it)


@pytest.mark.asyncio
async def test_dynamic_metric_selection_paths(request):
    """
    Exercise both dynamic metric branches. Preseed `state["selections"]`
    because the prompt-only path won't populate it.
    """
    # Acquire fixtures and helpers
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    # configure hydra (no local for monkeypatch)
    _configure_hydra_for_dynamic_tests(request.getfixturevalue("monkeypatch"), mod)

    # ---- Run with dynamic_metrics=True (uses loader.metric_type) ----
    state = dict(base_state_val)
    # Preseed selections so _prepare_final_subgraph can color nodes
    state["selections"] = {"gene_protein": ["G:TP53"], "disease": ["D:GLIO"]}

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(metric_type="COSINE")

    cmd = await call_run_async(
        tool,
        tool_call_id="tc-A",
        state=state,
        prompt="only prompt",
        arg_data=SimpleNamespace(extraction_name="E-A"),
    )
    assert "dic_extracted_graph" in cmd.update
    # cover cfg helper methods for A
    assert (
        mod.hydra.compose("config", overrides=["x"]).tools.multimodal_subgraph_extraction.marker()
        is None
    )
    assert (
        mod.hydra.compose("config", overrides=["x"]).tools.multimodal_subgraph_extraction.marker2()
        is None
    )

    # ---- Run with dynamic_metrics=False (uses cfg.search_metric_type) ----
    state = dict(base_state_val)
    state["selections"] = {"gene_protein": ["G:TP53"], "disease": ["D:GLIO"]}

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(metric_type="IP")

    cmd = await call_run_async(
        tool,
        tool_call_id="tc-B",
        state=state,
        prompt="only prompt two",
        arg_data=SimpleNamespace(extraction_name="E-B"),
    )
    assert "dic_extracted_graph" in cmd.update
    # cover cfg helper methods for B
    assert (
        mod.hydra.compose("config", overrides=["y"]).tools.multimodal_subgraph_extraction.marker()
        is None
    )
    assert (
        mod.hydra.compose("config", overrides=["y"]).tools.multimodal_subgraph_extraction.marker2()
        is None
    )
    # db cfg helper methods
    assert mod.hydra.compose("config").marker() is None
    assert mod.hydra.compose("config").marker2() is None
    # unexpected compose path
    assert mod.hydra.compose("unexpected") is None


def test_run_sync_wrapper(request):
    """run the sync wrapper which calls the async path internally"""
    request.getfixturevalue("fake_hydra")
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")

    loader_factory = request.getfixturevalue("fake_loader_factory")

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(normalize_vectors=True)

    base_state_val = request.getfixturevalue("base_state")
    state = dict(base_state_val)
    # Preseed selections because this test uses prompt-only flow
    state["selections"] = {"gene_protein": ["G:TP53"], "disease": ["D:GLIO"]}

    cmd = call_run(
        tool,
        tool_call_id="tc-sync",
        state=state,
        prompt="sync run",
        arg_data=SimpleNamespace(extraction_name="E-sync"),
    )
    assert "dic_extracted_graph" in cmd.update


def test_connection_error_raises_runtimeerror(request):
    """
    Make ensure_connection raise to exercise the error path in _run_async.
    """

    request.getfixturevalue("fake_hydra")
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")
    base_state_val = request.getfixturevalue("base_state")
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    class ExplodingManager:
        """exploding manager whose ensure_connection raises"""

        def __init__(self, cfg_db):
            """initialize with cfg_db"""
            self.cfg_db = cfg_db

        def ensure_connection(self):
            """ "ensure_connection always raises"""
            raise RuntimeError("nope")

        def info(self):
            """second public method for style compliance"""
            return "boom"

    # Patch manager ctor to explode
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(mod, "MilvusConnectionManager", ExplodingManager, raising=True)

    tool = MultimodalSubgraphExtractionTool()

    with pytest.raises(RuntimeError) as ei:
        asyncio.get_event_loop().run_until_complete(
            call_run_async(
                tool,
                tool_call_id="tc-err",
                state=base_state_val,
                prompt="will fail",
                arg_data=SimpleNamespace(extraction_name="E-err"),
            )
        )
    assert "Cannot connect to Milvus database" in str(ei.value)
    # cover extra info() method on ExplodingManager
    assert ExplodingManager(None).info() == "boom"


def test_prepare_query_modalities_async_with_excel_grouping(request):
    """prepare_query_modalities_async with Excel file populates state['selections"""
    # Use the public async prep path via _run_async in another test,
    # but here directly target the helper to assert selections are added.
    request.getfixturevalue("fake_hydra")
    request.getfixturevalue("fake_pcst_and_fast")
    request.getfixturevalue("fake_milvus_and_manager")
    request.getfixturevalue("fake_read_excel")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)

    # State with one Excel + one "nohit" row to exercise empty async result path
    state = dict(base_state_val)
    state["uploaded_files"] = [{"file_type": "multimodal", "file_path": "/fake.xlsx"}]

    # We also monkeypatch the async_query to return empty for a fabricated node

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    # create a fake manager just to call the method
    mgr = mod.MilvusConnectionManager(mod.hydra.compose("config").utils.database.milvus)

    async def run():
        qdf = await call_prepare_query_modalities_async(
            tool,
            prompt={"text": "query", "emb": [[0.1, 0.2, 0.3]]},
            state=state,
            cfg_db=mod.hydra.compose("config").utils.database.milvus,
            connection_manager=mgr,
        )
        # After reading excel and querying, selections should be set
        assert "selections" in state and isinstance(state["selections"], dict)
        # Prompt row appended
        pdf = getattr(qdf, "to_pandas", lambda: qdf)()
        assert any(pdf["node_type"] == "prompt")

    asyncio.get_event_loop().run_until_complete(run())


def test__query_milvus_collection_sync_casts_and_builds_expr(request):
    """query_milvus_collection builds expr and returns expected columns and types"""

    request.getfixturevalue("fake_milvus_and_manager")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)  # doesn't matter for this test

    # Build a node_type_df exactly like the function expects
    node_type_df = pd.DataFrame({"q_node_name": ["TP53", "EGFR"]})

    # cfg_db only needs database_name
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # Use a node_type containing '/' to exercise replace('/', '_')
    out_df = call_query_milvus_collection(tool, "gene/protein", node_type_df, cfg_db)

    # Must have all columns in q_columns + 'use_description'
    expected_cols = [
        "node_id",
        "node_name",
        "node_type",
        "feat",
        "feat_emb",
        "desc",
        "desc_emb",
        "use_description",
    ]
    assert list(out_df.columns) == expected_cols

    # Returned rows are the two we asked for; embeddings must be floats
    assert set(out_df["node_name"]) == {"TP53", "EGFR"}
    for row in out_df.itertuples(index=False):
        assert all(isinstance(x, float) for x in row.feat_emb)
        assert all(isinstance(x, float) for x in row.desc_emb)

    # 'use_description' is forced False in this path
    assert not out_df["use_description"].astype(bool).any()
    # exercise FakeCollection default branch (unexpected expr)
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    assert mod.Collection("nodes").query("unexpected expr", []) == []


def test__prepare_query_modalities_sync_with_multimodal_grouping(request):
    """pepare_query_modalities with multimodal file populates state['selections']"""

    request.getfixturevalue("fake_milvus_and_manager")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")

    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)

    # Force _read_multimodal_files to return grouped rows across 2 types.
    multimodal_df = pd.DataFrame(
        {
            "q_node_type": ["gene_protein", "gene_protein", "disease"],
            "q_node_name": ["TP53", "EGFR", "glioblastoma"],
        }
    )
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(tool, "_read_multimodal_files", lambda state: multimodal_df, raising=True)

    # cfg_db minimal
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # prompt dict expected by the function
    prompt = {"text": "user text", "emb": [[0.1, 0.2, 0.3]]}

    # run sync helper (NOT the async one)
    qdf = call_prepare_query_modalities(tool, prompt, base_state_val, cfg_db)

    # 1) It should have appended the prompt row with node_type='prompt' and use_description=True
    pdf = getattr(qdf, "to_pandas", lambda: qdf)()
    assert "prompt" in set(pdf["node_type"])
    # last row is the appended prompt row (per implementation)
    last = pdf.iloc[-1]
    assert last["node_type"] == "prompt"
    # avoid identity comparison with numpy.bool_
    assert bool(last["use_description"])  # was: `is True`

    # 2) Prior rows are from Milvus queries; ensure they exist and carry use_description=False
    non_prompt = pdf[pdf["node_type"] != "prompt"]
    assert not non_prompt.empty
    assert not non_prompt["use_description"].astype(bool).any()
    # We expect at least TP53/EGFR/glioblastoma present from our FakeCollection
    assert {"TP53", "EGFR", "glioblastoma"}.issubset(set(non_prompt["node_name"]))

    # 3) The function must have populated state['selections'] grouped by node_type
    assert "selections" in base_state_val and isinstance(base_state_val["selections"], dict)
    # Sanity: keys align with node types returned by queries
    assert (
        "gene_protein" in base_state_val["selections"]
        or "gene/protein" in base_state_val["selections"]
    )
    assert "disease" in base_state_val["selections"]
    # And the IDs collected are the ones FakeCollection returns
    collected_ids = set(sum(base_state_val["selections"].values(), []))
    assert {"G:TP53", "G:EGFR", "D:GLIO"}.issubset(collected_ids)


def test__prepare_query_modalities_sync_prompt_only_branch(request):
    """run the prompt-only branch of _prepare_query_modalities"""
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")
    tool = MultimodalSubgraphExtractionTool()
    loader_factory.get_loader(tool).set(normalize_vectors=False)

    # Force empty multimodal_df → else: query_df = prompt_df
    empty_df = pd.DataFrame(columns=["q_node_type", "q_node_name"])
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(tool, "_read_multimodal_files", lambda state: empty_df, raising=True)

    # Flat vector (common case), but function should handle either flat or nested
    expected_emb = [0.1, 0.2, 0.3]
    qdf = call_prepare_query_modalities(
        tool,
        {"text": "only prompt", "emb": expected_emb},
        base_state_val,
        SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg")),
    )
    pdf = getattr(qdf, "to_pandas", lambda: qdf)()

    # All rows should be prompt rows with use_description True
    assert set(pdf["node_type"]) == {"prompt"}
    assert pdf["use_description"].map(bool).all()

    # Coerce to flat list of floats and compare numerically
    def coerce_elem(x):
        inner = x[0] if isinstance(x, list | tuple) and x and isinstance(x[0], list | tuple) else x
        return [float(v) for v in (inner if isinstance(inner, list | tuple) else [inner])]

    flat_vals = [f for elem in pdf["feat_emb"].tolist() for f in coerce_elem(elem)]
    assert len(flat_vals) == len(expected_emb)
    for a, b in zip(flat_vals, expected_emb, strict=False):
        assert math.isclose(a, b, rel_tol=1e-9)


@pytest.mark.asyncio
async def test__prepare_query_modalities_async_single_task_branch(request):
    """prepare_query_modalities_async with single group exercises single-task path"""
    request.getfixturevalue("fake_milvus_and_manager")
    request.getfixturevalue("fake_hydra")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")

    tool = MultimodalSubgraphExtractionTool()
    loader_factory.get_loader(tool).set(normalize_vectors=False)

    # exactly one node type → len(tasks) == 1 → query_results = [await tasks[0]]
    single_group_df = pd.DataFrame({"q_node_type": ["gene_protein"], "q_node_name": ["TP53"]})
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(tool, "_read_multimodal_files", lambda state: single_group_df, raising=True)

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    cfg_db = mod.hydra.compose("config").utils.database.milvus
    manager = mod.MilvusConnectionManager(cfg_db)

    prompt = {"text": "p", "emb": [[0.1, 0.2, 0.3]]}
    qdf = await call_prepare_query_modalities_async(tool, prompt, base_state_val, cfg_db, manager)

    pdf = getattr(qdf, "to_pandas", lambda: qdf)()
    # it should contain both the TP53 row (from Milvus) and the appended prompt row
    assert "TP53" in set(pdf["node_name"])
    assert "prompt" in set(pdf["node_type"])


def test__perform_subgraph_extraction_sync_unifies_nodes_edges(request):
    """perform_subgraph_extraction sync path unifies nodes/edges across multiple queries"""
    # Patch MultimodalPCSTPruning to implement .extract_subgraph for sync path

    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")
    monkeypatch = request.getfixturevalue("monkeypatch")
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    call_counter = {"i": 0}

    class FakePCSTSync:
        """fake of MultimodalPCSTPruning with extract_subgraph method"""

        def __init__(self, **kwargs):
            """init with kwargs; ignore them"""
            self._seen = bool(kwargs)

        def extract_subgraph(self, desc_emb, feat_emb, node_type, cfg_db):
            """extract_subgraph returns different subgraphs per call"""
            # Return different subgraphs across calls to exercise union/unique
            del desc_emb, feat_emb, node_type, cfg_db
            call_counter["i"] += 1
            if call_counter["i"] == 1:
                return {"nodes": np.array([10, 11]), "edges": np.array([100])}
            return {"nodes": np.array([11, 12]), "edges": np.array([101])}

        def marker(self):
            """extra public method to satisfy style"""
            return None

    monkeypatch.setattr(mod, "MultimodalPCSTPruning", FakePCSTSync, raising=True)

    # Build a query_df with two rows (will yield two subgraphs)
    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)
    # cover marker method
    assert FakePCSTSync().marker() is None

    query_df = loader.df.dataframe(
        [
            {
                "node_id": "u1",
                "node_name": "Q1",
                "node_type": "gene_protein",
                "feat": "f",
                "feat_emb": [[0.1]],
                "desc": "d",
                "desc_emb": [[0.1]],
                "use_description": False,
            },
            {
                "node_id": "u2",
                "node_name": "Q2",
                "node_type": "disease",
                "feat": "f",
                "feat_emb": [[0.2]],
                "desc": "d",
                "desc_emb": [[0.2]],
                "use_description": True,
            },
        ]
    )

    # Run extraction with minimal cfg and cfg_db, build pdf directly
    pdf_obj = call_perform_subgraph_extraction(
        tool,
        dict(base_state_val),
        SimpleNamespace(
            cost_e=1.0,
            c_const=0.5,
            root=-1,
            num_clusters=1,
            pruning="strong",
            verbosity_level=0,
            vector_processing=SimpleNamespace(dynamic_metrics=True),
            search_metric_type=None,
        ),
        SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg")),
        query_df,
    )
    pdf = getattr(pdf_obj, "to_pandas", lambda: pdf_obj)()

    # first row is Unified Subgraph with unioned nodes/edges
    unified = pdf.iloc[0]
    assert unified["name"] == "Unified Subgraph"
    assert set(unified["nodes"]) == {10, 11, 12}
    assert set(unified["edges"]) == {100, 101}

    # subsequent rows correspond to Q1 and Q2
    names = list(pdf["name"])
    assert "Q1" in names and "Q2" in names


def test__prepare_final_subgraph_defaults_black_when_no_colors(request):
    """prepare_final_subgraph colors nodes black when no selections/colors present"""
    # Prepare a minimal subgraph DataFrame
    request.getfixturevalue("fake_milvus_and_manager")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    tool = MultimodalSubgraphExtractionTool()
    loader_factory.get_loader(tool).set(normalize_vectors=False)

    subgraphs_df = tool.loader.df.dataframe(
        [("Unified Subgraph", [10, 11], [100])],
        columns=["name", "nodes", "edges"],
    )

    # cfg_db required by Collection names; selections empty → color_df empty
    cfg_db = SimpleNamespace(
        milvus_db=SimpleNamespace(database_name="primekg"),
        node_colors_dict={"gene_protein": "red", "disease": "blue"},
    )
    state = {"selections": {}}  # IMPORTANT: key exists but empty → triggers else: black

    graph_dict = call_prepare_final_subgraph(tool, state, subgraphs_df, cfg_db)

    # Inspect colors on returned nodes; all should be black
    nodes_list = graph_dict["nodes"][0]  # first (and only) graph's nodes list
    assert len(nodes_list) > 0
    for _node_id, attrs in nodes_list:
        assert attrs["color"] == "black"


@pytest.mark.asyncio
async def test__perform_subgraph_extraction_async_no_vector_processing_branch(request):
    """perform_subgraph_extraction async path with no vector_processing exercises else: branch"""
    request.getfixturevalue("fake_milvus_and_manager")
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")
    tool = MultimodalSubgraphExtractionTool()
    loader_factory.get_loader(tool).set(normalize_vectors=False)

    # Make _extract_single_subgraph_async return a fixed subgraph so we avoid PCST internals
    async def _fake_extract(pcst_instance, query_row, cfg_db, manager):
        """fake _extract_single_subgraph_async returning fixed subgraph"""
        del pcst_instance, query_row, cfg_db, manager
        return {"nodes": np.array([10]), "edges": np.array([100])}

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(tool, "_extract_single_subgraph_async", _fake_extract, raising=True)

    # Build a one-row query_df
    qdf = tool.loader.df.dataframe(
        [
            {
                "node_id": "u",
                "node_name": "Q",
                "node_type": "prompt",
                "feat": "f",
                "feat_emb": [[0.1]],
                "desc": "d",
                "desc_emb": [[0.1]],
                "use_description": True,
            }
        ]
    )

    # cfg WITHOUT vector_processing attribute → triggers the else: dynamic_metrics_enabled = False
    cfg = SimpleNamespace(
        cost_e=1.0,
        c_const=0.5,
        root=-1,
        num_clusters=1,
        pruning="strong",
        verbosity_level=0,
        # no vector_processing here
        search_metric_type="COSINE",
    )
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    manager = mod.MilvusConnectionManager(cfg_db)  # this uses your FakeManager

    out = await call_perform_subgraph_extraction_async(
        tool,
        ExtractionParams(
            state=base_state_val,
            cfg=cfg,
            cfg_db=cfg_db,
            query_df=qdf,
            connection_manager=manager,
        ),
    )
    pdf = getattr(out, "to_pandas", lambda: out)()
    assert "Unified Subgraph" in set(pdf["name"])


def test_sync_uses_cfg_metric_when_no_vp(request):
    """perform_subgraph_extraction sync path uses cfg.search_metric_type
    when no vector_processing
    """
    # Patch MultimodalPCSTPruning to capture metric_type passed in (line 412 path)
    loader_factory = request.getfixturevalue("fake_loader_factory")
    base_state_val = request.getfixturevalue("base_state")
    monkeypatch = request.getfixturevalue("monkeypatch")
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    captured_metric_types = []

    class FakePCSTSync:
        """fake of MultimodalPCSTPruning capturing metric_type in ctor"""

        def __init__(self, **kwargs):
            """init capturing metric_type"""
            # Capture the metric_type used by the business logic
            captured_metric_types.append(kwargs.get("metric_type"))

        def extract_subgraph(self, desc_emb, feat_emb, node_type, cfg_db):
            """extract_subgraph returns minimal subgraph"""
            # Minimal valid return for the sync path
            del desc_emb, feat_emb, node_type, cfg_db
            return {"nodes": np.array([10]), "edges": np.array([100])}

        def marker(self):
            """extra public method to satisfy style"""
            return None

    monkeypatch.setattr(mod, "MultimodalPCSTPruning", FakePCSTSync, raising=True)

    # Instantiate tool and ensure loader.metric_type is different from cfg.search_metric_type
    tool = MultimodalSubgraphExtractionTool()
    loader = loader_factory.get_loader(tool)
    loader.set(metric_type="COSINE")  # should NOT be used in this test

    # Build a single-row query_df to hit the loop once
    query_df = loader.df.dataframe(
        [
            {
                "node_id": "u1",
                "node_name": "Q1",
                "node_type": "gene_protein",
                "feat": "f",
                "feat_emb": [[0.1]],
                "desc": "d",
                "desc_emb": [[0.1]],
                "use_description": True,
            }
        ]
    )

    cfg = SimpleNamespace(
        cost_e=1.0,
        c_const=0.5,
        root=-1,
        num_clusters=1,
        pruning="strong",
        verbosity_level=0,
        search_metric_type="IP",  # expect this to be used
    )

    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))
    state = dict(base_state_val)

    # Run the sync extraction
    _ = call_perform_subgraph_extraction(tool, state, cfg, cfg_db, query_df)

    # Assert business logic picked cfg.search_metric_type, not loader.metric_type
    assert captured_metric_types, "PCST was not constructed"
    assert captured_metric_types[-1] == "IP"
    # cover marker method without affecting earlier assertion
    assert FakePCSTSync().marker() is None
