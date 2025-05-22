"""
Unit tests for QAToolHelper routines in tool_helper.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper import QAToolHelper


class TestQAToolHelper(unittest.TestCase):
    """tests for QAToolHelper routines in tool_helper.py"""

    def setUp(self):
        """set up test case"""
        self.helper = QAToolHelper()

    def test_start_call_sets_config_and_call_id(self):
        """test start_call sets config and call_id"""
        cfg = SimpleNamespace(foo="bar")
        self.helper.start_call(cfg, "call123")
        self.assertIs(self.helper.config, cfg)
        self.assertEqual(self.helper.call_id, "call123")

    def test_init_vector_store_reuse(self):
        """test init_vector_store reuses existing instance"""
        emb_model = MagicMock()
        first = self.helper.init_vector_store(emb_model)
        second = self.helper.init_vector_store(emb_model)
        self.assertIs(second, first)

    def test_get_state_models_and_data_success(self):
        """test get_state_models_and_data returns models and data"""
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
        """test get_state_models_and_data raises ValueError if missing text embedding"""
        state = {"llm_model": MagicMock(), "article_data": {"p": {}}}
        with self.assertRaises(ValueError) as cm:
            self.helper.get_state_models_and_data(state)
        self.assertEqual(str(cm.exception), "No text embedding model found in state.")

    def test_get_state_models_and_data_missing_llm(self):
        """test get_state_models_and_data raises ValueError if missing LLM"""
        state = {"text_embedding_model": MagicMock(), "article_data": {"p": {}}}
        with self.assertRaises(ValueError) as cm:
            self.helper.get_state_models_and_data(state)
        self.assertEqual(str(cm.exception), "No LLM model found in state.")

    def test_get_state_models_and_data_missing_article_data(self):
        """test get_state_models_and_data raises ValueError if missing article data"""
        state = {"text_embedding_model": MagicMock(), "llm_model": MagicMock()}
        with self.assertRaises(ValueError) as cm:
            self.helper.get_state_models_and_data(state)
        self.assertEqual(str(cm.exception), "No article_data found in state.")

    def test_load_candidate_papers_calls_add_paper_only_for_valid(self):
        """test load_candidate_papers calls add_paper only for valid candidates"""
        vs = SimpleNamespace(loaded_papers=set(), add_paper=MagicMock())
        articles = {"p1": {"pdf_url": "url1"}, "p2": {}, "p3": {"pdf_url": None}}
        candidates = ["p1", "p2", "p3"]
        self.helper.load_candidate_papers(vs, articles, candidates)
        vs.add_paper.assert_called_once_with("p1", "url1", articles["p1"])

    def test_load_candidate_papers_handles_add_paper_exception(self):
        """test load_candidate_papers handles add_paper exception"""
        # If add_paper raises, it should be caught and not propagate
        vs = SimpleNamespace(
            loaded_papers=set(), add_paper=MagicMock(side_effect=ValueError("oops"))
        )
        articles = {"p1": {"pdf_url": "url1"}}
        # Start call to set call_id (used in logging)
        self.helper.start_call(SimpleNamespace(), "call001")
        # Should not raise despite exception
        self.helper.load_candidate_papers(vs, articles, ["p1"])
        vs.add_paper.assert_called_once_with("p1", "url1", articles["p1"])

    def test_run_reranker_success_and_filtering(self):
        """test run_reranker success and filtering"""
        # Successful rerank returns filtered candidates
        cfg = SimpleNamespace(top_k_papers=2)
        self.helper.config = cfg
        vs = MagicMock()
        with patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper.rank_papers_by_query",
            return_value=["a", "c"],
        ):
            out = self.helper.run_reranker(vs, "q", ["a", "b"])
        self.assertEqual(out, ["a"])

    def test_run_reranker_exception_fallback(self):
        """test run_reranker exception fallback"""
        # On reranker failure, should return original candidates
        cfg = SimpleNamespace(top_k_papers=5)
        self.helper.config = cfg
        vs = MagicMock()

        def fail(*args, **kwargs):
            raise RuntimeError("fail")

        with patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper.rank_papers_by_query",
            side_effect=fail,
        ):
            candidates = ["x", "y"]
            out = self.helper.run_reranker(vs, "q", candidates)
        self.assertEqual(out, candidates)

    def test_format_answer_with_and_without_sources(self):
        """test format_answer with and without sources"""
        articles = {"p1": {"Title": "T1"}, "p2": {"Title": "T2"}}
        # With sources
        with patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper.generate_answer",
            return_value={"output_text": "ans", "papers_used": ["p1", "p2"]},
        ):
            res = self.helper.format_answer("q", [], MagicMock(), articles)
            self.assertIn("ans", res)
            self.assertIn("Sources:", res)
            self.assertIn("- T1", res)
            self.assertIn("- T2", res)
        # Without sources
        with patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper.generate_answer",
            return_value={"output_text": "ans", "papers_used": []},
        ):
            res2 = self.helper.format_answer("q", [], MagicMock(), {})
            self.assertEqual(res2, "ans")
