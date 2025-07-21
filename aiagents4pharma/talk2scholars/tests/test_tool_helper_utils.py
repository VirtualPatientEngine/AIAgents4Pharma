"""
Unit tests for QAToolHelper routines in tool_helper.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper import QAToolHelper


class TestQAToolHelper(unittest.TestCase):
    """tests for QAToolHelper routines"""

    def setUp(self):
        """setup for each test"""
        self.helper = QAToolHelper()

    def test_start_call_sets_config_and_call_id(self):
        """start_call should set config and call_id"""
        cfg = SimpleNamespace(foo="bar")
        self.helper.start_call(cfg, "call123")
        self.assertIs(self.helper.config, cfg)
        self.assertEqual(self.helper.call_id, "call123")

    @patch("aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper.get_vectorstore")
    def test_init_vector_store_reuse(self, mock_get_vectorstore):
        """Mock vector store reuse test"""
        emb_model = MagicMock()
        mock_instance = MagicMock()
        mock_get_vectorstore.return_value = mock_instance

        first = self.helper.init_vector_store(emb_model)
        second = self.helper.init_vector_store(emb_model)

        self.assertIs(first, second)
        self.assertIs(second, mock_instance)

    def test_get_state_models_and_data_success(self):
        """get_state_models_and_data should return models and data from state"""
        emb = MagicMock()
        llm = MagicMock()
        articles = {"p": {}}
        state = {
            "text_embedding_model": emb,
            "llm_model": llm,
            "article_data": articles,
        }
        ret_emb, ret_llm, ret_articles = self.helper.get_state_models_and_data(state)
        self.assertIs(ret_emb, emb)
        self.assertIs(ret_llm, llm)
        self.assertIs(ret_articles, articles)

    def test_get_state_models_and_data_missing_text_embedding(self):
        """get_state_models_and_data should raise ValueError if text_embedding_model is missing"""
        state = {"llm_model": MagicMock(), "article_data": {"p": {}}}
        with self.assertRaises(ValueError):
            self.helper.get_state_models_and_data(state)

    def test_get_state_models_and_data_missing_llm(self):
        """should raise ValueError if llm_model is missing"""
        state = {"text_embedding_model": MagicMock(), "article_data": {"p": {}}}
        with self.assertRaises(ValueError):
            self.helper.get_state_models_and_data(state)

    def test_get_state_models_and_data_missing_article_data(self):
        """get_state_models_and_data should raise ValueError if article_data is missing"""
        state = {"text_embedding_model": MagicMock(), "llm_model": MagicMock()}
        with self.assertRaises(ValueError):
            self.helper.get_state_models_and_data(state)

    def test_get_hardware_stats(self):
        """get_hardware_stats should return correct GPU and hardware mode"""
        helper = QAToolHelper()
        helper.call_id = "test_call"

        helper.has_gpu = False
        stats = helper.get_hardware_stats()
        self.assertEqual(stats["gpu_available"], False)
        self.assertEqual(stats["hardware_mode"], "CPU-only")

        helper.has_gpu = True
        stats = helper.get_hardware_stats()
        self.assertEqual(stats["gpu_available"], True)
        self.assertEqual(stats["hardware_mode"], "GPU-accelerated")
