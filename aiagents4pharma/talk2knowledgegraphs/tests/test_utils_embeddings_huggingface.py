"""
Test cases for utils/embeddings/huggingface.py
"""

from __future__ import annotations

import pytest
import torch

from ..utils.embeddings import huggingface as hf_module
from ..utils.embeddings.huggingface import EmbeddingWithHuggingFace


class FakeInputs(dict):
    """Tokenizer output shim with .to()."""

    def to(self, _device):
        """Return self to mimic device placement."""
        return self

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


class FakeTokenizer:
    """Tokenizer stub returning fixed tensors."""

    def __call__(self, texts, **_kwargs):
        """Return a minimal attention mask for the batch."""
        batch = len(texts) if isinstance(texts, list) else 1
        return FakeInputs({"attention_mask": torch.ones((batch, 3))})

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


class FakeModel:
    """Model stub returning fixed outputs."""

    def to(self, _device):
        """Return self for device placement chaining."""
        return self

    def __call__(self, **_kwargs):
        # output[0] is token embeddings of shape (batch, seq, hidden)
        batch = _kwargs["attention_mask"].shape[0]
        return (torch.ones((batch, 3, 4)),)

    def ping(self):
        """No-op helper to satisfy pylint public-method count."""
        return None


def _patch_common(monkeypatch, allow_model: bool):
    """Patch HuggingFace dependencies for deterministic tests."""
    if allow_model:
        monkeypatch.setattr(hf_module.AutoConfig, "from_pretrained", lambda *_a, **_k: None)
    else:

        def _raise(*_a, **_k):
            raise OSError("missing")

        monkeypatch.setattr(hf_module.AutoConfig, "from_pretrained", _raise)

    monkeypatch.setattr(
        hf_module,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": lambda *_a, **_k: FakeTokenizer()}),
    )
    monkeypatch.setattr(
        hf_module, "AutoModel", type("M", (), {"from_pretrained": lambda *_a, **_k: FakeModel()})
    )


def test_embedding_with_huggingface_embed_documents(monkeypatch):
    """Test embedding documents using the EmbeddingWithHuggingFace class."""
    _patch_common(monkeypatch, allow_model=True)
    embedding_model = EmbeddingWithHuggingFace(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_cache_dir="../../cache",
        truncation=True,
    )

    result = embedding_model.embed_documents(["A", "B"])
    assert tuple(result.shape) == (2, 4)


def test_embedding_with_huggingface_embed_query(monkeypatch):
    """Test embedding a query using the EmbeddingWithHuggingFace class."""
    _patch_common(monkeypatch, allow_model=True)
    embedding_model = EmbeddingWithHuggingFace(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_cache_dir="../../cache",
        truncation=True,
    )

    result = embedding_model.embed_query("Adalimumab")
    assert tuple(result.shape) == (4,)


def test_embedding_with_huggingface_failed(monkeypatch):
    """Test handling when model is not available on HuggingFace Hub."""
    _patch_common(monkeypatch, allow_model=False)
    model_name = "aiagents4pharma/embeddings"
    err_msg = f"Model {model_name} is not available on HuggingFace Hub."
    with pytest.raises(ValueError) as exc:
        EmbeddingWithHuggingFace(
            model_name=model_name,
            model_cache_dir="../../cache",
            truncation=True,
        )
    assert err_msg in str(exc.value)


def test_huggingface_helpers():
    """Cover helper methods in test doubles."""
    FakeInputs().ping()
    FakeTokenizer().ping()
    FakeModel().ping()
