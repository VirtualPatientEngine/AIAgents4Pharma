"""
Unit tests for Zotero human in the loop in zotero_review.py with structured output.
"""

import unittest
from unittest.mock import patch, MagicMock
from aiagents4pharma.talk2scholars.tools.zotero.zotero_review import zotero_review


class TestZoteroReviewTool(unittest.TestCase):
    """Test class for Zotero review tool with structured LLM output."""

    def setUp(self):
        self.tool_call_id = "tc"
        self.collection_path = "/Col"
        # Create a sample fetched papers dictionary with one paper.
        self.sample_papers = {"p1": {"Title": "T1", "Authors": ["A1", "A2", "A3"]}}

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value=None,
    )
    def test_no_fetched_papers(self, mock_fetch):
        """Test when no fetched papers are found."""
        with self.assertRaises(ValueError) as context:
            zotero_review.run(
                {
                    "tool_call_id": self.tool_call_id,
                    "collection_path": self.collection_path,
                    "state": {},
                }
            )
        self.assertIn("No fetched papers were found to save", str(context.exception))
        mock_fetch.assert_called_once()

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1", "A2"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value="dummy_response",
    )
    def test_missing_llm_model(self, mock_interrupt, mock_fetch):
        """Test when LLM model is not available in state, expecting fallback confirmation."""
        state = {"last_displayed_papers": self.sample_papers}  # llm_model missing
        result = zotero_review.run(
            {
                "tool_call_id": self.tool_call_id,
                "collection_path": self.collection_path,
                "state": state,
            }
        )
        upd = result.update
        # The fallback message should start with "REVIEW REQUIRED:"
        self.assertTrue(upd["messages"][0].content.startswith("REVIEW REQUIRED:"))
        # Check that the approval status is set as fallback values.
        approval = upd["zotero_write_approval_status"]
        self.assertEqual(approval["collection_path"], self.collection_path)
        self.assertTrue(approval["papers_reviewed"])
        self.assertFalse(approval["approved"])
        self.assertEqual(approval["papers_count"], len(self.sample_papers))
        mock_fetch.assert_called_once()
        mock_interrupt.assert_called_once()

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1", "A2"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value="dummy_response",
    )
    def test_human_approve(self, mock_interrupt, mock_fetch):
        """Test when human approves saving papers using structured output."""
        # Prepare a fake llm_model with structured output.
        fake_structured_llm = MagicMock()
        # Simulate invoke() returns an object with decision "approve"
        fake_decision = MagicMock()
        fake_decision.decision = "approve"
        fake_structured_llm.invoke.return_value = fake_decision

        fake_llm_model = MagicMock()
        fake_llm_model.with_structured_output.return_value = fake_structured_llm

        state = {
            "last_displayed_papers": self.sample_papers,
            "llm_model": fake_llm_model,
        }

        result = zotero_review.run(
            {
                "tool_call_id": self.tool_call_id,
                "collection_path": self.collection_path,
                "state": state,
            }
        )

        upd = result.update
        self.assertEqual(
            upd["zotero_write_approval_status"],
            {"collection_path": self.collection_path, "approved": True},
        )
        self.assertIn(
            f"Human approved saving 1 papers to Zotero collection '{self.collection_path}'",
            upd["messages"][0].content,
        )
        mock_fetch.assert_called_once()
        mock_interrupt.assert_called_once()

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1", "A2"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value="dummy_response",
    )
    def test_human_approve_custom(self, mock_interrupt, mock_fetch):
        """Test when human approves with a custom collection path."""
        fake_structured_llm = MagicMock()
        fake_decision = MagicMock()
        fake_decision.decision = "custom"
        fake_decision.custom_path = "/Custom"
        fake_structured_llm.invoke.return_value = fake_decision

        fake_llm_model = MagicMock()
        fake_llm_model.with_structured_output.return_value = fake_structured_llm

        state = {
            "last_displayed_papers": self.sample_papers,
            "llm_model": fake_llm_model,
        }

        result = zotero_review.run(
            {
                "tool_call_id": self.tool_call_id,
                "collection_path": self.collection_path,
                "state": state,
            }
        )

        upd = result.update
        self.assertEqual(
            upd["zotero_write_approval_status"],
            {"collection_path": "/Custom", "approved": True},
        )
        self.assertIn(
            "Human approved saving papers to custom Zotero collection '/Custom'",
            upd["messages"][0].content,
        )
        mock_fetch.assert_called_once()
        mock_interrupt.assert_called_once()

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={"p1": {"Title": "T1", "Authors": ["A1", "A2"]}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value="dummy_response",
    )
    def test_human_reject(self, mock_interrupt, mock_fetch):
        """Test when human rejects saving papers via structured output."""
        fake_structured_llm = MagicMock()
        fake_decision = MagicMock()
        fake_decision.decision = "reject"
        fake_structured_llm.invoke.return_value = fake_decision

        fake_llm_model = MagicMock()
        fake_llm_model.with_structured_output.return_value = fake_structured_llm

        state = {
            "last_displayed_papers": self.sample_papers,
            "llm_model": fake_llm_model,
        }

        result = zotero_review.run(
            {
                "tool_call_id": self.tool_call_id,
                "collection_path": self.collection_path,
                "state": state,
            }
        )

        upd = result.update
        self.assertEqual(upd["zotero_write_approval_status"], {"approved": False})
        self.assertIn(
            "Human rejected saving papers to Zotero", upd["messages"][0].content
        )
        mock_fetch.assert_called_once()
        mock_interrupt.assert_called_once()

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt")
    def test_structured_processing_failure(self, mock_interrupt, mock_fetch):
        """Test fallback when structured review processing fails."""
        # Simulate valid fetched papers with multiple entries.
        papers = {
            f"p{i}": {"Title": f"Title{i}", "Authors": [f"A{i}"]} for i in range(1, 8)
        }
        mock_fetch.return_value = papers
        mock_interrupt.return_value = "dummy_response"
        # Provide a fake llm_model whose invoke() raises an exception.
        fake_structured_llm = MagicMock()
        fake_structured_llm.invoke.side_effect = Exception("structured error")
        fake_llm_model = MagicMock()
        fake_llm_model.with_structured_output.return_value = fake_structured_llm

        state = {"last_displayed_papers": papers, "llm_model": fake_llm_model}

        result = zotero_review.run(
            {
                "tool_call_id": self.tool_call_id,
                "collection_path": "/MyCol",
                "state": state,
            }
        )

        upd = result.update
        content = upd["messages"][0].content
        # The fallback message should start with "REVIEW REQUIRED:".
        self.assertTrue(content.startswith("REVIEW REQUIRED:"))
        self.assertIn("Would you like to save 7 papers", content)
        self.assertIn("... and 2 more papers", content)

        approved = upd["zotero_write_approval_status"]
        self.assertEqual(approved["collection_path"], "/MyCol")
        self.assertTrue(approved["papers_reviewed"])
        self.assertFalse(approved["approved"])
        self.assertEqual(approved["papers_count"], 7)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.fetch_papers_for_save",
        return_value={
            "p1": {"Title": "Test Paper", "Authors": ["Alice", "Bob", "Charlie"]}
        },
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_review.interrupt",
        return_value="dummy_response",
    )
    def test_authors_et_al_in_summary(self, mock_interrupt, mock_fetch):
        """
        Test that the papers summary includes 'et al.' when a paper has more than two authors.
        This is achieved by forcing a fallback (structured processing failure) so that the fallback
        message with the papers summary is generated.
        """
        # Create a fake llm_model whose structured output processing fails.
        fake_structured_llm = MagicMock()
        fake_structured_llm.invoke.side_effect = Exception("structured error")
        fake_llm_model = MagicMock()
        fake_llm_model.with_structured_output.return_value = fake_structured_llm

        state = {
            "last_displayed_papers": {
                "p1": {"Title": "Test Paper", "Authors": ["Alice", "Bob", "Charlie"]}
            },
            "llm_model": fake_llm_model,
        }
        result = zotero_review.run(
            {
                "tool_call_id": self.tool_call_id,
                "collection_path": self.collection_path,
                "state": state,
            }
        )
        fallback_message = result.update["messages"][0].content
        self.assertIn("et al.", fallback_message)
        mock_fetch.assert_called_once()
        mock_interrupt.assert_called_once()
