"""
Unit tests for NVIDIA NIM reranker error handling in nvidia_nim_reranker.py
"""

import unittest
from types import SimpleNamespace

from aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker import (
    rank_papers_by_query,
)


class TestNVIDIARerankerError(unittest.TestCase):
    """Tests for NVIDIA NIM reranker error handling."""

    def test_missing_api_key_raises_value_error(self):
        """Ensure missing API key triggers ValueError."""
        vector_store = SimpleNamespace(documents={})
        # Config without API key
        cfg = SimpleNamespace(
            reranker=SimpleNamespace(model="m", api_key=None), top_k_papers=3
        )
        with self.assertRaises(ValueError) as cm:
            rank_papers_by_query(vector_store, "query", cfg, top_k=cfg.top_k_papers)
        self.assertEqual(
            str(cm.exception),
            "Configuration 'reranker.api_key' must be set for reranking",
        )
