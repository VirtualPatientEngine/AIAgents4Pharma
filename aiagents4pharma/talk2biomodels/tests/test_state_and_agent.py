"""
Unit-mock coverage for Talk2BioModels state and agent wiring.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..agents import t2b_agent
from ..states.state_talk2biomodels import Talk2Biomodels, add_data

pytestmark = pytest.mark.unit_mock


class FakeCtx:
    """Hydra initialize context manager stub."""

    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def test_state_and_add_data():
    """Verify state defaults and add_data merge behavior."""
    state = Talk2Biomodels(
        llm_model=MagicMock(),
        text_embedding_model=MagicMock(),
        pdf_file_name="file.pdf",
        model_id=[],
        sbml_file_path=[],
        model_as_string=[],
        dic_simulated_data=[],
        dic_scanned_data=[],
        dic_steady_state_data=[],
        dic_annotations_data=[],
    )
    assert state["pdf_file_name"] == "file.pdf"
    merged = add_data([{"name": "a", "val": 1}], [{"name": "a", "val": 2}])
    assert merged[0]["val"] == 2
    merged2 = add_data([{"name": "a", "val": 1}], [{"name": "b", "val": 3}])
    assert any(item["name"] == "b" for item in merged2)


def test_get_app(monkeypatch):
    """Cover agent wiring and graph compilation."""
    dummy_cfg = SimpleNamespace(
        agents=SimpleNamespace(t2b_agent=SimpleNamespace(state_modifier="prompt"))
    )

    class DummyGraph:
        """Minimal StateGraph stub."""

        def __init__(self, *_args, **_kwargs):
            self.nodes = {}

        def add_node(self, name, fn):
            """Register a node."""
            self.nodes[name] = fn

        def add_edge(self, *_args, **_kwargs):
            """No-op edge registration."""
            return None

        def compile(self, **_kwargs):
            """Trigger first node for coverage and return sentinel."""
            if self.nodes:
                list(self.nodes.values())[0]({"messages": []})
            return "compiled"

    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.agents.t2b_agent.hydra.initialize",
        lambda **_kwargs: FakeCtx(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.agents.t2b_agent.hydra.compose",
        lambda **_kwargs: dummy_cfg,
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.agents.t2b_agent.create_react_agent",
        lambda *_args, **_kwargs: MagicMock(invoke=lambda *_a, **_k: {}),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.agents.t2b_agent.StateGraph",
        lambda *_args, **_kwargs: DummyGraph(),
    )
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.agents.t2b_agent.MemorySaver",
        lambda *_args, **_kwargs: MagicMock(),
    )
    assert t2b_agent.get_app(1, llm_model=MagicMock()) == "compiled"
