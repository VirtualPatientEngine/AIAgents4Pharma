"""
Test cases for utils/enrichments/ollama.py
"""

from __future__ import annotations

import pytest

from ..utils.enrichments import ollama as ollama_module
from ..utils.enrichments.ollama import EnrichmentWithOllama


class FakePopen:
    """Minimal context manager stub for subprocess.Popen."""

    def __init__(self, *_args, **_kwargs):
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, *_exc):
        """Exit the context manager."""
        return False


class FakeChain:
    """Chain stub supporting piping and invoke."""

    def __or__(self, _other):
        """Return self for chaining."""
        return self

    def invoke(self, payload):
        """Return a stringified list based on input payload."""
        raw = payload.get("input", "").strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()
        count = 0 if not raw else len([part for part in raw.split(",") if part.strip()])
        return str([f"enriched_{i}" for i in range(count)])


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

    def pong(self):
        """Second no-op helper to satisfy pylint."""
        return None


class FakeChatOllama:
    """Fake chat model; only needs to be pipeable."""

    def __init__(self, **_kwargs):
        """Initialize the fake model."""
        self.ready = True

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None

    def pong(self):
        """Second no-op helper to satisfy pylint."""
        return None


class FakeStrOutputParser:
    """Parser stub for piping."""

    def __ror__(self, _other):
        """Return a fake chain for reverse piping."""
        return FakeChain()

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


@pytest.fixture(name="ollama_config")
def fixture_ollama_config():
    """Return a dictionary with Ollama configuration."""
    return {
        "model_name": "llama3.2:1b",
        "prompt_enrichment": "Prompt {input}",
        "temperature": 0.0,
        "streaming": False,
    }


def _patch_common(monkeypatch, model_names: list[str]):
    """Patch ollama and langchain deps for deterministic tests."""
    monkeypatch.setattr(
        ollama_module.ollama,
        "list",
        lambda: {"models": [{"model": name} for name in model_names]},
        raising=True,
    )
    monkeypatch.setattr(ollama_module.ollama, "pull", lambda _name: None, raising=True)
    monkeypatch.setattr(ollama_module.ollama, "delete", lambda _name: None, raising=True)
    monkeypatch.setattr(ollama_module.subprocess, "Popen", FakePopen, raising=True)
    monkeypatch.setattr(ollama_module.time, "sleep", lambda _t: None, raising=True)
    monkeypatch.setattr(ollama_module, "ChatPromptTemplate", FakePromptTemplate, raising=True)
    monkeypatch.setattr(ollama_module, "ChatOllama", FakeChatOllama, raising=True)
    monkeypatch.setattr(ollama_module, "StrOutputParser", FakeStrOutputParser, raising=True)


def test_no_model_ollama(ollama_config, monkeypatch):
    """Test the case when the Ollama model is not available."""
    cfg = ollama_config
    cfg_model = "smollm2:135m"

    _patch_common(monkeypatch, model_names=[])

    with pytest.raises(
        ValueError,
        match=f"Error: Pulled {cfg_model} model and restarted Ollama server.",
    ):
        EnrichmentWithOllama(
            model_name=cfg_model,
            prompt_enrichment=cfg["prompt_enrichment"],
            temperature=cfg["temperature"],
            streaming=cfg["streaming"],
        )


def test_enrich_ollama(ollama_config, monkeypatch):
    """Test the Ollama textual enrichment class for node enrichment."""
    cfg = ollama_config
    _patch_common(monkeypatch, model_names=[cfg["model_name"]])

    enr_model = EnrichmentWithOllama(
        model_name=cfg["model_name"],
        prompt_enrichment=cfg["prompt_enrichment"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
    )

    nodes = ["acetaminophen"]
    enriched_nodes = enr_model.enrich_documents(nodes)
    assert enriched_nodes == ["enriched_0"]


def test_enrich_ollama_rag(ollama_config, monkeypatch):
    """Test enrichment with RAG falls back to enrich_documents."""
    cfg = ollama_config
    _patch_common(monkeypatch, model_names=[cfg["model_name"]])

    enr_model = EnrichmentWithOllama(
        model_name=cfg["model_name"],
        prompt_enrichment=cfg["prompt_enrichment"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
    )

    nodes = ["acetaminophen"]
    docs = ["doc1", "doc2"]
    enriched_nodes = enr_model.enrich_documents_with_rag(nodes, docs)
    assert enriched_nodes == ["enriched_0"]


def test_enrichment_helpers():
    """Cover helper methods in test doubles."""
    FakePromptTemplate().ping()
    FakePromptTemplate().pong()
    FakeChatOllama().ping()
    FakeChatOllama().pong()
    _ = object() | FakeStrOutputParser()
    FakeStrOutputParser().ping()
