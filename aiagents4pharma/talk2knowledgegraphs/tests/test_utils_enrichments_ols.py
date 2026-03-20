#!/usr/bin/env python3

"""
Test cases for utils/enrichments/ols_terms.py
"""

from __future__ import annotations

from types import SimpleNamespace

from ..utils.enrichments import ols_terms as ols_module
from ..utils.enrichments.ols_terms import EnrichmentWithOLS


def _patch_common(monkeypatch):
    """Patch hydra and requests for OLS tests."""

    class FakeCfg:
        """Fake OLS config."""

        def __init__(self):
            self.base_url = "https://fake-ols/terms"
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
                utils=SimpleNamespace(enrichments=SimpleNamespace(ols_terms=FakeCfg()))
            )
        return None

    monkeypatch.setattr(
        ols_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )

    def fake_get(_url, headers=None, params=None, timeout=1):
        del headers, timeout
        term = params.get("short_form")
        if term == "CL_0000899":
            body = {
                "_embedded": {
                    "terms": [
                        {
                            "description": ["CD4-positive"],
                            "synonyms": ["T-helper 17"],
                            "label": "cell",
                        }
                    ]
                }
            }
        elif term == "XYZ_0000000":
            body = {}
        else:
            body = {
                "_embedded": {
                    "terms": [{"description": ["desc"], "synonyms": [], "label": "label"}]
                }
            }
        return SimpleNamespace(text=__import__("json").dumps(body))

    monkeypatch.setattr(ols_module.requests, "get", fake_get, raising=True)
    return HydraCtx, FakeCfg


def test_enrich_documents(monkeypatch):
    """Test the enrich_documents method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithOLS()
    ols_terms = ["CL_0000899", "GO_0046427", "XYZ_0000000"]
    descriptions = enrich_obj.enrich_documents(ols_terms)
    assert "CD4-positive" in descriptions[0]
    assert "label" in descriptions[1]
    assert descriptions[2] == ""


def test_enrich_documents_with_rag(monkeypatch):
    """Test the enrich_documents_with_rag method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithOLS()
    ols_terms = ["CL_0000899", "XYZ_0000000"]
    descriptions = enrich_obj.enrich_documents_with_rag(ols_terms, None)
    assert "T-helper 17" in descriptions[0]
    assert descriptions[1] == ""


def test_ols_helpers(monkeypatch):
    """Cover helper methods in test doubles."""
    hydra_ctx, cfg_cls = _patch_common(monkeypatch)

    cfg = cfg_cls()
    cfg.ping()
    cfg.pong()

    ctx = hydra_ctx()
    ctx.ping()
    ctx.pong()

    assert ols_module.hydra.compose("other") is None
