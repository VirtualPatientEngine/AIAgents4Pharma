"""
Test cases for tools/multimodal_subgraph_extraction.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from ..tools import multimodal_subgraph_extraction as multimodal_module
from ..tools.multimodal_subgraph_extraction import MultimodalSubgraphExtractionTool

DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"


def _fake_graph():
    pyg_graph = Data(
        x=[[0.1, 0.2], [0.3, 0.4]],
        node_id=["A", "B"],
        node_name=["A", "B"],
        node_type=["gene/protein", "disease"],
        desc_x=np.array([[0.1, 0.2], [0.2, 0.3]]),
        enriched_node=[False, True],
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
        edge_type=["rel", "rel"],
        enriched_edge=[False, False],
    )
    text_graph = {
        "nodes": pd.Series(["node_id,node_attr\nA,descA", "node_id,node_attr\nB,descB"]),
        "edges": pd.Series(
            [
                "head_id,edge_type,tail_id\nA,('gene','rel','disease'),B",
                "head_id,edge_type,tail_id\nB,('disease','rel','gene'),A",
            ]
        ),
    }
    return pyg_graph, text_graph


class FakeEmbedding:
    """Fake embedding wrapper for Ollama embeddings."""

    def __init__(self, model_name: str):
        """Initialize with a model name."""
        self.model_name = model_name

    def embed_query(self, _prompt: str):
        """Return a deterministic embedding."""
        return [0.1, 0.2]

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


class FakePCST:
    """Fake PCST pruning implementation."""

    def __init__(self, **_kwargs):
        """Initialize the fake PCST."""
        self.ready = True

    def extract_subgraph(self, _graph, _desc, _feat, _modality):
        """Return a fixed subgraph."""
        return {"nodes": np.array([0, 1]), "edges": np.array([0])}

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


def _patch_common(monkeypatch):
    """Patch hydra, embeddings, pcst, and joblib for deterministic tests."""

    class FakeCfg:
        """Fake config for multimodal subgraph extraction."""

        def __init__(self):
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.ollama_embeddings = ["fake-model"]
            self.node_colors_dict = {"gene/protein": "#111111", "disease": "#222222"}

        def ping(self):
            """No-op helper to satisfy pylint."""
            return None

        def pong(self):
            """Second no-op helper to satisfy pylint."""
            return None

    class HydraCtx:
        """Hydra context manager stub."""

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def ping(self):
            """No-op helper to satisfy pylint."""
            return None

        def pong(self):
            """Second no-op helper to satisfy pylint."""
            return None

    def initialize(**_kwargs):
        return HydraCtx()

    def compose(config_name, overrides=None):
        del overrides
        if config_name == "config":
            return SimpleNamespace(tools=SimpleNamespace(multimodal_subgraph_extraction=FakeCfg()))
        return None

    monkeypatch.setattr(
        multimodal_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    monkeypatch.setattr(multimodal_module, "EmbeddingWithOllama", FakeEmbedding, raising=True)
    monkeypatch.setattr(multimodal_module, "MultimodalPCSTPruning", FakePCST, raising=True)

    pyg_graph, text_graph = _fake_graph()

    def fake_load(path):
        """Return a fake graph based on the file path."""
        if path.endswith("pyg_graph.pkl"):
            return pyg_graph
        return text_graph

    monkeypatch.setattr(multimodal_module.joblib, "load", fake_load, raising=True)
    return HydraCtx, FakeCfg


@pytest.fixture(name="agent_state")
def agent_state_fixture():
    """Provide a minimal agent state for multimodal tests."""
    return {
        "selections": {
            "gene/protein": [],
            "molecular_function": [],
            "cellular_component": [],
            "biological_process": [],
            "drug": [],
            "disease": [],
        },
        "uploaded_files": [],
        "topk_nodes": 3,
        "topk_edges": 3,
        "dic_source_graph": [
            {
                "name": "BioBridge",
                "kg_pyg_path": f"{DATA_PATH}/biobridge_multimodal_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/biobridge_multimodal_text_graph.pkl",
            }
        ],
    }


def test_extract_multimodal_subgraph_wo_doc(agent_state, tool_call, monkeypatch):
    """Test the multimodal subgraph extraction tool for only text as modality."""
    _patch_common(monkeypatch)

    tool = MultimodalSubgraphExtractionTool()
    response = tool.invoke(
        tool_call(
            tool,
            {
                "prompt": "Extract IBD subgraph.",
                "state": agent_state,
                "arg_data": {"extraction_name": "subkg_12345"},
            },
        )
    )

    dic_extracted_graph = response.update["dic_extracted_graph"][0]
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "BioBridge"
    assert len(dic_extracted_graph["graph_dict"]["nodes"]) > 0
    assert len(dic_extracted_graph["graph_dict"]["edges"]) > 0


def test_multimodal_helper_methods(monkeypatch):
    """Cover helper methods in test doubles."""
    hydra_ctx, cfg_cls = _patch_common(monkeypatch)

    FakeEmbedding("fake").ping()
    FakePCST().ping()

    cfg = cfg_cls()
    cfg.ping()
    cfg.pong()

    ctx = hydra_ctx()
    ctx.ping()
    ctx.pong()

    assert multimodal_module.hydra.compose("other") is None


def test_extract_multimodal_subgraph_w_doc(agent_state, tool_call, monkeypatch):
    """Test the multimodal subgraph extraction tool for text as modality, plus genes."""
    _patch_common(monkeypatch)

    def fake_read_excel(path, sheet_name=None):
        """Return a fake multimodal sheet mapping."""
        del path, sheet_name
        return {"gene-protein": pd.DataFrame({"name": ["A"]})}

    monkeypatch.setattr(pd, "read_excel", fake_read_excel, raising=True)

    agent_state["uploaded_files"] = [
        {
            "file_name": "multimodal-analysis.xlsx",
            "file_path": f"{DATA_PATH}/multimodal-analysis.xlsx",
            "file_type": "multimodal",
            "uploaded_by": "VPEUser",
            "uploaded_timestamp": "2025-05-12 00:00:00",
        }
    ]

    tool = MultimodalSubgraphExtractionTool()
    response = tool.invoke(
        tool_call(
            tool,
            {
                "prompt": "Extract IBD subgraph.",
                "state": agent_state,
                "arg_data": {"extraction_name": "subkg_12345"},
            },
        )
    )

    dic_extracted_graph = response.update["dic_extracted_graph"][0]
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "BioBridge"
    assert len(dic_extracted_graph["graph_dict"]["nodes"]) > 0
    assert len(dic_extracted_graph["graph_dict"]["edges"]) > 0
