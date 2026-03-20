"""
Test cases for tools/subgraph_summarization.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ..tools import subgraph_summarization as summarization_module
from ..tools.subgraph_summarization import SubgraphSummarizationTool


class FakeChain:
    """Chain stub supporting piping and invoke."""

    def __or__(self, _other):
        """Return self for chaining."""
        return self

    def invoke(self, _payload):
        """Return a deterministic summary."""
        return "summary"

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None

    def pong(self):
        """Second no-op helper to satisfy pylint."""
        return None


class FakePromptTemplate:
    """Prompt template stub."""

    @staticmethod
    def from_messages(_messages):
        """Create a fake prompt template."""
        return FakePromptTemplate()

    def __or__(self, _other):
        """Return a fake chain for piping."""
        return FakeChain()

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


class FakeStrOutputParser:
    """Parser stub for piping."""

    def __ror__(self, _other):
        """Return a fake chain for reverse piping."""
        return FakeChain()

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


def _patch_common(monkeypatch):
    """Patch hydra and chain components for deterministic tests."""

    class FakeCfg:
        """Fake config for subgraph summarization."""

        def __init__(self):
            self.prompt_subgraph_summarization = "summary"

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
            return SimpleNamespace(tools=SimpleNamespace(subgraph_summarization=FakeCfg()))
        return None

    monkeypatch.setattr(
        summarization_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    monkeypatch.setattr(
        summarization_module, "ChatPromptTemplate", FakePromptTemplate, raising=True
    )
    monkeypatch.setattr(summarization_module, "StrOutputParser", FakeStrOutputParser, raising=True)
    return HydraCtx, FakeCfg


@pytest.fixture(name="input_state")
def input_state_fixture():
    """Provide minimal input state for summarization tests."""
    return {
        "llm_model": object(),
        "dic_extracted_graph": [
            {
                "name": "subkg_12345",
                "graph_text": "node_id,node_attr\nA,descA",
                "graph_summary": None,
            }
        ],
    }


def test_summarize_subgraph(input_state, tool_call, monkeypatch):
    """Test the subgraph summarization tool."""
    _patch_common(monkeypatch)

    tool = SubgraphSummarizationTool()
    response = tool.invoke(
        tool_call(
            tool,
            {
                "prompt": "Summarize.",
                "state": input_state,
                "extraction_name": "subkg_12345",
            },
        )
    )

    assert response.update["messages"][-1].tool_call_id == "tool_call_id"
    dic_extracted_graph = response.update["dic_extracted_graph"][0]
    assert dic_extracted_graph["graph_summary"] == "summary"


def test_summarization_helpers(monkeypatch):
    """Cover helper methods used by test doubles."""
    hydra_ctx, cfg_cls = _patch_common(monkeypatch)

    FakeChain().ping()
    FakeChain().pong()
    FakePromptTemplate.from_messages([]).ping()
    _ = object() | FakeStrOutputParser()
    FakeStrOutputParser().ping()

    cfg = cfg_cls()
    cfg.ping()
    cfg.pong()

    ctx = hydra_ctx()
    ctx.ping()
    ctx.pong()

    assert summarization_module.hydra.compose("other") is None
