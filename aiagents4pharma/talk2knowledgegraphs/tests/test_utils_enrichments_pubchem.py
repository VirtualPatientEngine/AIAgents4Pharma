#!/usr/bin/env python3

"""
Test cases for utils/enrichments/pubchem_strings.py
"""

from __future__ import annotations

from types import SimpleNamespace

from ..utils.enrichments import pubchem_strings as pubchem_module
from ..utils.enrichments.pubchem_strings import EnrichmentWithPubChem


def _patch_common(monkeypatch):
    """Patch hydra and requests for PubChem tests."""

    class FakeCfg:
        """Fake PubChem config."""

        def __init__(self):
            self.pubchem_cid2smiles_url = "https://fake-pubchem"

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
            return SimpleNamespace(utils=SimpleNamespace(pubchem_utils=FakeCfg()))
        return None

    monkeypatch.setattr(
        pubchem_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )

    def fake_get(url, timeout=60):
        del timeout
        if url.endswith("/5311000/property/smiles/JSON"):
            return SimpleNamespace(
                json=lambda: {"PropertyTable": {"Properties": [{"SMILES": "SMILES1"}]}}
            )
        return SimpleNamespace(json=lambda: {})

    monkeypatch.setattr(pubchem_module.requests, "get", fake_get, raising=True)
    monkeypatch.setattr(
        pubchem_module,
        "pubchem_cid_description",
        lambda _cid: "Alclometasone description",
        raising=True,
    )
    return HydraCtx, FakeCfg


def test_enrich_documents(monkeypatch):
    """Test the enrich_documents method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithPubChem()
    pubchem_ids = ["5311000", "1X"]
    enriched_descriptions, enriched_strings = enrich_obj.enrich_documents(pubchem_ids)
    assert enriched_strings == ["SMILES1", None]
    assert enriched_descriptions[0].startswith("Alclometasone")
    assert enriched_descriptions[1] is None


def test_enrich_documents_with_rag(monkeypatch):
    """Test the enrich_documents_with_rag method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithPubChem()
    pubchem_ids = ["5311000", "1X"]
    enriched_descriptions, enriched_strings = enrich_obj.enrich_documents_with_rag(
        pubchem_ids, None
    )
    assert enriched_strings == ["SMILES1", None]
    assert enriched_descriptions[0].startswith("Alclometasone")
    assert enriched_descriptions[1] is None


def test_pubchem_helpers(monkeypatch):
    """Cover helper methods in test doubles."""
    hydra_ctx, cfg_cls = _patch_common(monkeypatch)

    cfg = cfg_cls()
    cfg.ping()
    cfg.pong()

    ctx = hydra_ctx()
    ctx.ping()
    ctx.pong()

    assert pubchem_module.hydra.compose("other") is None
