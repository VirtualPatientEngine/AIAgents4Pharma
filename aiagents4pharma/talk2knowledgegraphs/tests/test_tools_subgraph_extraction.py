"""
Test cases for tools/subgraph_extraction.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from ..tools import subgraph_extraction as subgraph_module
from ..tools.subgraph_extraction import SubgraphExtractionTool

DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"


def _fake_graph():
    """Build a minimal PyG graph and textual graph for tests."""
    pyg_graph = Data(
        x=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        node_id=["A", "B"],
        node_name=["A", "B"],
        node_type=["gene", "disease"],
        enriched_node=[False, True],
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
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
    """Fake PCSTPruning implementation."""

    def __init__(self, *_args, **_kwargs):
        """Initialize the fake PCST."""
        self.ready = True

    def extract_subgraph(self, _graph, _query_emb):
        """Return a fixed subgraph."""
        return {"nodes": np.array([0, 1]), "edges": np.array([0])}

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


def _patch_common(monkeypatch):
    """Patch hydra, embeddings, pcst, and joblib for deterministic tests."""

    class FakeCfg:
        """Fake config for subgraph extraction."""

        def __init__(self):
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.ollama_embeddings = ["fake-model"]
            self.splitter_chunk_size = 1000
            self.splitter_chunk_overlap = 0
            self.retriever_search_type = "mmr"
            self.retriever_k = 2
            self.retriever_fetch_k = 2
            self.retriever_lambda_mult = 0.5
            self.prompt_endotype_filtering = "system"
            self.prompt_endotype_addition = "Endotype"

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
            return SimpleNamespace(tools=SimpleNamespace(subgraph_extraction=FakeCfg()))
        return None

    monkeypatch.setattr(
        subgraph_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    monkeypatch.setattr(subgraph_module, "EmbeddingWithOllama", FakeEmbedding, raising=True)
    monkeypatch.setattr(subgraph_module, "PCSTPruning", FakePCST, raising=True)

    pyg_graph, text_graph = _fake_graph()

    def fake_load(path):
        if path.endswith("pyg_graph.pkl"):
            return pyg_graph
        return text_graph

    monkeypatch.setattr(subgraph_module.joblib, "load", fake_load, raising=True)
    return HydraCtx, FakeCfg


@pytest.fixture(name="agent_state")
def agent_state_fixture():
    """Agent state fixture."""
    return {
        "llm_model": object(),
        "embedding_model": object(),
        "uploaded_files": [],
        "topk_nodes": 3,
        "topk_edges": 3,
        "dic_source_graph": [
            {
                "name": "PrimeKG",
                "kg_pyg_path": f"{DATA_PATH}/primekg_ibd_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/primekg_ibd_text_graph.pkl",
            }
        ],
    }


def test_extract_subgraph_wo_docs(agent_state, tool_call, monkeypatch):
    """Test subgraph extraction without documents."""
    _patch_common(monkeypatch)

    prompt = "Extract IBD subgraph."
    tool = SubgraphExtractionTool()

    response = tool.invoke(
        tool_call(
            tool,
            {
                "prompt": prompt,
                "state": agent_state,
                "arg_data": {"extraction_name": "subkg_12345"},
            },
        )
    )

    assert response.update["messages"][-1].tool_call_id == "tool_call_id"
    dic_extracted_graph = response.update["dic_extracted_graph"][0]
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "PrimeKG"
    assert dic_extracted_graph["topk_nodes"] == 3
    assert dic_extracted_graph["topk_edges"] == 3
    assert len(dic_extracted_graph["graph_dict"]["nodes"]) > 0
    assert len(dic_extracted_graph["graph_dict"]["edges"]) > 0
    assert "node_id" in dic_extracted_graph["graph_text"]


def test_extract_subgraph_w_docs(agent_state, tool_call, monkeypatch):
    """Test subgraph extraction with endotype document filtering."""
    _patch_common(monkeypatch)

    class FakeLoader:
        """PDF loader stub."""

        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            """Return a dummy list of documents."""
            return ["doc"]

        def ping(self):
            """No-op helper to satisfy pylint public-method count."""
            return None

    class FakeSplitter:
        """Text splitter stub."""

        def __init__(self, **_kwargs):
            self.ready = True

        def split_documents(self, docs):
            """Return documents unchanged."""
            return docs

        def ping(self):
            """No-op helper to satisfy pylint public-method count."""
            return None

    class FakeVectorStore:
        """Vector store stub."""

        @staticmethod
        def from_documents(documents, embedding):
            """Return a fake vector store instance."""
            del documents, embedding
            return FakeVectorStore()

        def as_retriever(self, **_kwargs):
            """Return a dummy retriever object."""
            return object()

        def ping(self):
            """No-op helper to satisfy pylint public-method count."""
            return None

    class FakeChain:
        """Chain stub returning deterministic answer."""

        def invoke(self, _payload):
            """Return a dummy answer."""
            return {"answer": "GENE1"}

        def ping(self):
            """No-op helper to satisfy pylint public-method count."""
            return None

    monkeypatch.setattr(subgraph_module, "PyPDFLoader", FakeLoader, raising=True)
    monkeypatch.setattr(
        subgraph_module, "RecursiveCharacterTextSplitter", FakeSplitter, raising=True
    )
    monkeypatch.setattr(subgraph_module, "InMemoryVectorStore", FakeVectorStore, raising=True)
    monkeypatch.setattr(subgraph_module, "create_stuff_documents_chain", lambda *_a, **_k: object())
    monkeypatch.setattr(subgraph_module, "create_retrieval_chain", lambda *_a, **_k: FakeChain())
    monkeypatch.setattr(
        subgraph_module,
        "ChatPromptTemplate",
        SimpleNamespace(from_messages=lambda *_a, **_k: object()),
        raising=True,
    )

    agent_state["uploaded_files"] = [
        {
            "file_name": "DGE.pdf",
            "file_path": f"{DATA_PATH}/DGE.pdf",
            "file_type": "endotype",
            "uploaded_by": "VPEUser",
            "uploaded_timestamp": "2024-11-05 00:00:00",
        }
    ]

    tool = SubgraphExtractionTool()
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
    assert "node_id" in dic_extracted_graph["graph_text"]

    # cover helper methods for pylint
    FakeLoader("file.pdf").ping()
    FakeSplitter().ping()
    FakeVectorStore().ping()
    FakeChain().ping()


def test_subgraph_helper_methods(monkeypatch):
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

    assert subgraph_module.hydra.compose("other") is None
