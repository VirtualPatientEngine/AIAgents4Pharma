#!/usr/bin/env python3

"""
Test cases for utils/enrichments/uniprot_proteins.py
"""

from __future__ import annotations

from types import SimpleNamespace

from ..utils.enrichments import uniprot_proteins as uniprot_module
from ..utils.enrichments.uniprot_proteins import EnrichmentWithUniProt


def _patch_common(monkeypatch):
    """Patch hydra and requests for UniProt tests."""

    class FakeCfg:
        """Fake UniProt config."""

        def __init__(self):
            self.reviewed = True
            self.isoform = False
            self.organism = 9606
            self.uniprot_url = "https://fake-uniprot"
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
                utils=SimpleNamespace(enrichments=SimpleNamespace(uniprot_proteins=FakeCfg()))
            )
        return None

    monkeypatch.setattr(
        uniprot_module,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )

    def fake_get(_url, headers=None, params=None, timeout=1):
        del headers, timeout
        gene = params.get("exact_gene")
        if gene == "TP53":
            body = [
                {
                    "comments": [
                        {
                            "type": "FUNCTION",
                            "text": [{"value": "Multifunctional transcription factor"}],
                        }
                    ],
                    "sequence": {"sequence": "MEEPQSDPSV"},
                }
            ]
            return SimpleNamespace(ok=True, text=__import__("json").dumps(body))
        if gene == "TP5":
            return SimpleNamespace(ok=True, text=__import__("json").dumps([]))
        return SimpleNamespace(ok=False, text="")

    monkeypatch.setattr(uniprot_module.requests, "get", fake_get, raising=True)
    return HydraCtx, FakeCfg


def test_enrich_documents(monkeypatch):
    """Test the enrich_documents method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithUniProt()
    gene_names = ["TP53", "TP5", "XZ"]
    descriptions, sequences = enrich_obj.enrich_documents(gene_names)
    assert descriptions[0].startswith("Multifunctional transcription factor")
    assert sequences[0].startswith("MEEPQSDPSV")
    assert descriptions[1] is None
    assert sequences[1] is None
    assert descriptions[2] is None
    assert sequences[2] is None


def test_enrich_documents_with_rag(monkeypatch):
    """Test the enrich_documents_with_rag method."""
    _patch_common(monkeypatch)
    enrich_obj = EnrichmentWithUniProt()
    gene_names = ["TP53", "TP5", "XZ"]
    descriptions, sequences = enrich_obj.enrich_documents_with_rag(gene_names, None)
    assert descriptions[0].startswith("Multifunctional transcription factor")
    assert sequences[0].startswith("MEEPQSDPSV")
    assert descriptions[1] is None
    assert sequences[1] is None
    assert descriptions[2] is None
    assert sequences[2] is None


def test_uniprot_helpers(monkeypatch):
    """Cover helper methods in test doubles."""
    hydra_ctx, cfg_cls = _patch_common(monkeypatch)

    cfg = cfg_cls()
    cfg.ping()
    cfg.pong()

    ctx = hydra_ctx()
    ctx.ping()
    ctx.pong()

    assert uniprot_module.hydra.compose("other") is None
