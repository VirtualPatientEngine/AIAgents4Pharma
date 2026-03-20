"""
Test cases for utils/embeddings/ollama.py
"""

from __future__ import annotations

import pytest

from ..utils.embeddings import ollama as ollama_module
from ..utils.embeddings.ollama import EmbeddingWithOllama


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

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None

    def pong(self):
        """Second no-op helper to satisfy pylint."""
        return None


class FakeOllamaEmbeddings:
    """Fake embeddings backend returning deterministic vectors."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return a fixed embedding for each input text."""
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, _text: str) -> list[float]:
        """Return a fixed embedding for a query."""
        return [0.1, 0.2, 0.3]


@pytest.fixture(name="ollama_config")
def fixture_ollama_config():
    """Return a dictionary with Ollama configuration."""
    return {
        "model_name": "all-minilm",
    }


def _patch_common(monkeypatch, model_names: list[str]):
    """Patch ollama and subprocess/time dependencies for deterministic tests."""

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
    monkeypatch.setattr(ollama_module, "OllamaEmbeddings", FakeOllamaEmbeddings, raising=True)


def test_no_model_ollama(ollama_config, monkeypatch):
    """Test the case when the Ollama model is not available."""
    cfg = ollama_config

    _patch_common(monkeypatch, model_names=[])

    with pytest.raises(
        ValueError,
        match=f"Error: Pulled {cfg['model_name']} model and restarted Ollama server.",
    ):
        EmbeddingWithOllama(model_name=cfg["model_name"])


@pytest.fixture(name="embedding_model")
def embedding_model_fixture(ollama_config, monkeypatch):
    """Return the configuration object for the Ollama embedding model and model object."""
    cfg = ollama_config
    _patch_common(monkeypatch, model_names=[cfg["model_name"]])
    return EmbeddingWithOllama(model_name=cfg["model_name"])


def test_embedding_with_ollama_embed_documents(embedding_model):
    """Test embedding documents using the EmbeddingWithOllama class."""
    texts = ["Adalimumab", "Infliximab", "Vedolizumab"]
    result = embedding_model.embed_documents(texts)
    assert result == [[1.0, 2.0, 3.0]] * 3


def test_embedding_with_ollama_embed_query(embedding_model):
    """Test embedding a query using the EmbeddingWithOllama class."""
    text = "Adalimumab"
    result = embedding_model.embed_query(text)
    assert result == [0.1, 0.2, 0.3]


def test_ollama_popen_helpers():
    """Cover helper methods in FakePopen."""
    FakePopen().ping()
    FakePopen().pong()
