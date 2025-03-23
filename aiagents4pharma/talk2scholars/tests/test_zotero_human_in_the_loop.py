"""
Unit tests for Zotero human in the loop in zotero_review.py.
"""

import unittest
from unittest.mock import patch

from langgraph.types import Command
from aiagents4pharma.talk2scholars.tools.zotero.zotero_review import (
    zotero_review,
)


class TestZoteroReviewTool(unittest.TestCase):
    """test class for Zotero review tool"""

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value=None,
    )
    def test_no_fetched_papers(self, mock_fetch):
        """Test when no fetched papers are found"""
        result = zotero_review.run(
            {"tool_call_id": "tc", "collection_path": "/Col", "state": {}}
        )
        mock_fetch.assert_called_once()
        self.assertIsInstance(result, Command)
        self.assertIn(
            "No fetched papers were found to save", result.update["messages"][0].content
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value=True,
    )
    def test_human_approve_boolean(self, mock_interrupt, mock_fetch):
        """Test when human approves saving papers"""
        result = zotero_review.run(
            {
                "tool_call_id": "tc",
                "collection_path": "/Col",
                "state": {"last_displayed_papers": "dummy"},
            }
        )
        mock_fetch.return_value = {"p1": {"Title": "T1", "Authors": ["A1"]}}
        mock_interrupt.return_value = True
        upd = result.update
        self.assertEqual(
            upd["approved_zotero_save"], {"collection_path": "/Col", "approved": True}
        )
        self.assertIn(
            "Human approved saving 1 papers to Zotero collection '/Col'",
            upd["messages"][0].content,
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value={"custom_path": "/Custom"},
    )
    def test_human_approve_custom_path(self, mock_interrupt, mock_fetch):
        """Test when human approves saving papers to custom path"""
        result = zotero_review.run(
            {
                "tool_call_id": "tc",
                "collection_path": "/Col",
                "state": {"last_displayed_papers": "dummy"},
            }
        )
        mock_fetch.return_value = {"p1": {"Title": "T1", "Authors": ["A1"]}}
        mock_interrupt.return_value = {"custom_path": "/Custom"}
        upd = result.update
        self.assertEqual(
            upd["approved_zotero_save"],
            {"collection_path": "/Custom", "approved": True},
        )
        self.assertIn(
            "Human approved saving papers to custom Zotero collection '/Custom'",
            upd["messages"][0].content,
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value=False,
    )
    def test_human_reject(self, mock_interrupt, mock_fetch):
        """Test when human rejects saving papers"""
        result = zotero_review.run(
            {
                "tool_call_id": "tc",
                "collection_path": "/Col",
                "state": {"last_displayed_papers": "dummy"},
            }
        )
        mock_fetch.return_value = {"p1": {"Title": "T1", "Authors": ["A1"]}}
        mock_interrupt.return_value = False
        upd = result.update
        self.assertEqual(upd["approved_zotero_save"], {"approved": False})
        self.assertIn(
            "Human rejected saving papers to Zotero", upd["messages"][0].content
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        side_effect=Exception("no interrupt"),
    )
    def test_interrupt_exception_summary(self, mock_interrupt, mock_fetch):
        """Test when an exception is raised during interrupt"""
        papers = {
            f"id{i}": {"Title": f"Title{i}", "Authors": ["A1", "A2", "A3"]}
            for i in range(7)
        }
        mock_fetch.return_value = papers
        mock_interrupt.side_effect = Exception("no interrupt")

        result = zotero_review.run(
            {
                "tool_call_id": "tc",
                "collection_path": "/MyCol",
                "state": {"last_displayed_papers": "dummy"},
            }
        )
        upd = result.update
        content = upd["messages"][0].content

        self.assertTrue(content.startswith("REVIEW REQUIRED:"))
        self.assertIn("Would you like to save 7 papers", content)
        self.assertIn("... and 2 more papers", content)

        approved = upd["approved_zotero_save"]
        self.assertEqual(approved["collection_path"], "/MyCol")
        self.assertTrue(approved["papers_reviewed"])
        self.assertFalse(approved["approved"])
        self.assertEqual(approved["papers_count"], 7)
