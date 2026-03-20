#!/usr/bin/env python3

"""
Test cases for utils/enrichments/reactome_pathways.py
"""

from __future__ import annotations

from types import SimpleNamespace

from ..utils.enrichments import reactome_pathways as reactome_module
from ..utils.enrichments.reactome_pathways import EnrichmentWithReactome


def _patch_common(monkeypatch):
    """Patch hydra and requests for Reactome tests."""

    class FakeCfg:
        """Fake Reactome config."""

        def __init__(self):
            self.base_url = "https://fake-reactome/"
            self.timeout = 1

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
            return SimpleNamespace(
                utils=SimpleNamespace(enrichments=SimpleNamespace(reactome_pathways=FakeCfg()))
            )
        return None

    monkeypatch.setattr(
        reactome_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )

    def fake_get(url, headers=None, timeout=1):
        del headers, timeout
        if url.endswith("R-HSA-3244647/summation"):
            return SimpleNamespace(ok=True, text="x\tCyclic GMP-AMP summary")
        if url.endswith("R-HSA-9905952/summation"):
            return SimpleNamespace(ok=True, text="x\tThe P2RX7 summary")
        return SimpleNamespace(ok=False, text="")

    monkeypatch.setattr(reactome_module.requests, "get", fake_get, raising=True)
    return HydraCtx, FakeCfg


def test_enrich_documents(monkeypatch):
    """Test the enrich_documents method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithReactome()
    reactome_pathways = ["R-HSA-3244647", "R-HSA-9905952", "R-HSA-1234567"]
    descriptions = enrich_obj.enrich_documents(reactome_pathways)
    assert descriptions[0].startswith("Cyclic GMP-AMP")
    assert descriptions[1].startswith("The P2RX7")
    assert descriptions[2] is None


def test_enrich_documents_with_rag(monkeypatch):
    """Test the enrich_documents_with_rag method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithReactome()
    reactome_pathways = ["R-HSA-3244647", "R-HSA-9905952", "R-HSA-1234567"]
    descriptions = enrich_obj.enrich_documents_with_rag(reactome_pathways, None)
    assert descriptions[0].startswith("Cyclic GMP-AMP")
    assert descriptions[1].startswith("The P2RX7")
    assert descriptions[2] is None


def test_reactome_helpers(monkeypatch):
    """Cover helper methods in test doubles."""
    hydra_ctx, cfg_cls = _patch_common(monkeypatch)

    cfg = cfg_cls()
    cfg.ping()
    cfg.pong()

    ctx = hydra_ctx()
    ctx.ping()
    ctx.pong()

    assert reactome_module.hydra.compose("other") is None
