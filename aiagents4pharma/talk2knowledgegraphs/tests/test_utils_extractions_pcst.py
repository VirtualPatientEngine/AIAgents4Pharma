"""
Test cases for pcst pruning utilities.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ..utils.extractions import multimodal_pcst as multimodal_module
from ..utils.extractions import pcst as pcst_module
from ..utils.extractions.multimodal_pcst import MultimodalPCSTPruning
from ..utils.extractions.pcst import PCSTPruning


def _make_graph() -> Data:
    return Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        node_id=["A", "B"],
        node_type=["gene", "disease"],
        desc_x=torch.tensor([[0.2, 0.8], [0.3, 0.7]]),
        enriched_node=[False, False],
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        edge_type=["rel", "rel"],
        enriched_edge=[False, False],
    )


def test_pcst_pruning_costs_and_nodes_edges():
    """Validate cost computation and node/edge selection with virtual nodes."""
    graph = _make_graph()
    pruner = PCSTPruning(topk=2, topk_e=2, cost_e=0.5, c_const=0.1)
    prizes = {
        "nodes": torch.tensor([1.0, 0.5]),
        "edges": torch.tensor([0.9, 0.1]),
    }

    edges_dict, prize_vec, costs, mapping = pruner.compute_subgraph_costs(graph, prizes)

    assert edges_dict["num_prior_edges"] == 1
    assert mapping["nodes"]
    assert prize_vec.shape[0] > graph.num_nodes
    assert costs.size > 0

    vertices = np.array([0, graph.num_nodes])
    edges = np.array([0, 1])
    subgraph = pruner.get_subgraph_nodes_edges(
        graph,
        vertices,
        {"edges": edges, "num_prior_edges": edges_dict["num_prior_edges"]},
        mapping,
    )

    assert 0 in subgraph["nodes"]
    assert subgraph["edges"].size > 0


def test_pcst_extract_subgraph(monkeypatch):
    """Ensure extract_subgraph wires prizes, costs, and PCST output."""
    graph = _make_graph()
    pruner = PCSTPruning(topk=2, topk_e=2)
    query_emb = torch.tensor([1.0, 0.0])

    def fake_pcst(_edges, _prizes, _costs, *_args):
        return np.array([0, 1]), np.array([0])

    monkeypatch.setattr(pcst_module.pcst_fast, "pcst_fast", fake_pcst, raising=True)

    subgraph = pruner.extract_subgraph(graph, query_emb)

    assert set(subgraph["nodes"]) == {0, 1}
    assert isinstance(subgraph["edges"], list | np.ndarray)


def test_multimodal_prizes_use_description():
    """Compute prizes when textual descriptions are used."""
    graph = _make_graph()
    pruner = MultimodalPCSTPruning(topk=2, topk_e=2, use_description=True)
    text_emb = torch.tensor([1.0, 0.0])
    query_emb = torch.tensor([1.0, 0.0])

    prizes = pruner.compute_prizes(graph, text_emb, query_emb, modality="gene")

    assert prizes["nodes"].shape[0] == graph.num_nodes
    assert prizes["edges"].shape[0] == graph.edge_index.shape[1]


def test_multimodal_costs_and_virtual_edges():
    """Cover both edge-cost branches and virtual edge handling."""
    graph = _make_graph()
    pruner = MultimodalPCSTPruning(topk=2, topk_e=2, cost_e=0.5, c_const=0.1)

    prizes_low = {"nodes": torch.tensor([1.0, 0.5]), "edges": torch.tensor([0.0, 0.0])}
    edges_dict_low, _, costs_low, mapping_low = pruner.compute_subgraph_costs(graph, prizes_low)
    assert edges_dict_low["num_prior_edges"] > 0
    assert mapping_low["edges"]
    assert len(costs_low) > 0

    prizes_high = {"nodes": torch.tensor([1.0, 0.5]), "edges": torch.tensor([1.0, 1.0])}
    _, _, costs_high, mapping_high = pruner.compute_subgraph_costs(graph, prizes_high)
    assert mapping_high["nodes"]
    assert costs_high.size > 0

    vertices = np.array([0, graph.num_nodes])
    edges_dict_virtual = {"edges": [], "num_prior_edges": 0}
    mapping_virtual = {"nodes": {graph.num_nodes: 0}, "edges": {}}
    subgraph = pruner.get_subgraph_nodes_edges(graph, vertices, edges_dict_virtual, mapping_virtual)
    assert 0 in subgraph["nodes"]


def test_multimodal_extract_subgraph(monkeypatch):
    """Ensure multimodal extract_subgraph runs end-to-end."""
    graph = _make_graph()
    pruner = MultimodalPCSTPruning(topk=2, topk_e=2, use_description=False)
    text_emb = torch.tensor([1.0, 0.0])
    query_emb = torch.tensor([1.0, 0.0])

    def fake_pcst(_edges, _prizes, _costs, *_args):
        return np.array([0, 1]), np.array([0])

    monkeypatch.setattr(multimodal_module.pcst_fast, "pcst_fast", fake_pcst, raising=True)

    subgraph = pruner.extract_subgraph(graph, text_emb, query_emb, modality="gene")

    assert set(subgraph["nodes"]) == {0, 1}
    assert isinstance(subgraph["edges"], list | np.ndarray)
