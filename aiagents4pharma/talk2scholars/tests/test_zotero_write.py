"""
Unit tests for Zotero write tool in zotero_write.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from langgraph.types import Command

from aiagents4pharma.talk2scholars.tools.zotero.zotero_write import zotero_save_tool

dummy_zotero_write_config = SimpleNamespace(
    user_id="dummy", library_type="user", api_key="dummy"
)
dummy_cfg = SimpleNamespace(
    tools=SimpleNamespace(zotero_write=dummy_zotero_write_config)
)


class TestZoteroSaveTool(unittest.TestCase):
    def setUp(self):
        # Patch Hydra and Zotero client globally
        self.hydra_init = patch(
            "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize"
        ).start()
        self.hydra_compose = patch(
            "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose",
            return_value=dummy_cfg,
        ).start()
        self.zotero_class = patch(
            "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero"
        ).start()

        self.fake_zot = MagicMock()
        self.zotero_class.return_value = self.fake_zot

    def tearDown(self):
        patch.stopall()

    def make_state(self, papers=None, approved=True, path="/Test Collection"):
        state = {}
        if approved:
            state["approved_zotero_save"] = {"approved": True, "collection_path": path}
        if papers is not None:
            state["last_displayed_papers"] = (
                papers if isinstance(papers, dict) else "papers"
            )
            if isinstance(papers, dict):
                state["papers"] = papers
        return state

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.fetch_papers_for_save",
        return_value=None,
    )
    def test_no_papers_after_approval(self, mock_fetch):
        result = zotero_save_tool.run(
            {
                "tool_call_id": "id",
                "collection_path": "/Test Collection",
                "state": self.make_state({}, True),
            }
        )
        self.assertIn(
            "No fetched papers were found to save", result.update["messages"][0].content
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.find_or_create_collection",
        return_value=None,
    )
    def test_invalid_collection(self, mock_find, mock_fetch):
        self.fake_zot.collections.return_value = [
            {"key": "k1", "data": {"name": "Existing"}}
        ]
        result = zotero_save_tool.run(
            {
                "tool_call_id": "id",
                "collection_path": "/DoesNotExist",
                "state": self.make_state({"p1": {}}, True),
            }
        )
        content = result.update["messages"][0].content
        self.assertIn("Error: Collection path mismatch", content)
        self.assertIn("/DoesNotExist", content)
        self.assertIn("/Test Collection", content)
        self.assertIn(
            "Error: Collection path mismatch",
            result.update["messages"][0].content,
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.find_or_create_collection",
        return_value="colKey",
    )
    def test_save_failure(self, mock_find, mock_fetch):
        self.fake_zot.collections.return_value = [
            {"key": "colKey", "data": {"name": "Test Collection"}}
        ]
        self.fake_zot.create_items.side_effect = Exception("Creation error")
        result = zotero_save_tool.run(
            {
                "tool_call_id": "id",
                "collection_path": "/Test Collection",
                "state": self.make_state({"p1": {}}, True),
            }
        )
        self.assertIn(
            "Error saving papers to Zotero", result.update["messages"][0].content
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.find_or_create_collection",
        return_value="colKey",
    )
    def test_successful_save(self, mock_find, mock_fetch):
        self.fake_zot.collections.return_value = [
            {"key": "colKey", "data": {"name": "Test Collection"}}
        ]
        self.fake_zot.create_items.return_value = {
            "successful": {"0": {"key": "item1"}}
        }

        result = zotero_save_tool.run(
            {
                "tool_call_id": "id",
                "collection_path": "/Test Collection",
                "state": self.make_state({"p1": {}}, True),
            }
        )
        content = result.update["messages"][0].content
        self.assertIn("Save was successful", content)
        self.assertIn("Test Collection", content)

    def test_without_approval(self):
        result = zotero_save_tool.run(
            {"tool_call_id": "id", "collection_path": "/Test Collection", "state": {}}
        )
        self.assertIn("not reviewed by user", result.update["messages"][0].content)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.find_or_create_collection",
        return_value=None,
    )
    def test_invalid_collection_exists_branch(self, mock_find, mock_fetch):
        self.fake_zot.collections.return_value = [
            {"key": "k1", "data": {"name": "Existing"}}
        ]
        state = self.make_state({"p1": {}}, approved=True, path="/DoesNotExist")

        result = zotero_save_tool.run(
            {"tool_call_id": "id", "collection_path": "/DoesNotExist", "state": state}
        )
        content = result.update["messages"][0].content
        self.assertIn("does not exist in Zotero", content)
        self.assertIn("Existing", content)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.fetch_papers_for_save",
        return_value=None,
    )
    def test_user_confirms_via_text_then_no_papers(self, mock_fetch):
        """If user_confirmation is truthy & approved via text, we mark approved then hit no‑papers path."""
        state = {
            "approved_zotero_save": {
                "approved": False,
                "papers_reviewed": True,
                "collection_path": "/Test",
            }
        }
        result = zotero_save_tool.run(
            {
                "tool_call_id": "id",
                "collection_path": "/Test",
                "state": state,
                "user_confirmation": "Yes",
            }
        )
        content = result.update["messages"][0].content
        self.assertIn("No fetched papers were found to save", content)

    def test_user_rejects_via_text(self):
        """If user_confirmation is non‑empty but not an approval keyword, return rejected Command."""
        state = {
            "approved_zotero_save": {
                "approved": False,
                "papers_reviewed": True,
                "collection_path": "/Test",
            }
        }
        result = zotero_save_tool.run(
            {
                "tool_call_id": "id",
                "collection_path": "/Test",
                "state": state,
                "user_confirmation": "Nope",
            }
        )
        content = result.update["messages"][0].content
        self.assertIn("Save operation was rejected by the user", content)
        self.assertEqual(result.update.get("approved_zotero_save"), {"approved": False})

    def test_rejected_without_review(self):
        """If approval_info exists but no papers_reviewed flag, it’s rejected."""
        state = {"approved_zotero_save": {"approved": False}}
        result = zotero_save_tool.run(
            {"tool_call_id": "id", "collection_path": "/Test", "state": state}
        )
        content = result.update["messages"][0].content
        self.assertIn("Save operation was rejected by the user", content)

    def test_awaiting_user_confirmation(self):
        """If papers_reviewed=True but approved=False and no user_confirmation, ask for confirmation."""
        state = {
            "approved_zotero_save": {
                "approved": False,
                "papers_reviewed": True,
                "collection_path": "/Test",
            }
        }
        result = zotero_save_tool.run(
            {"tool_call_id": "id", "collection_path": "/Test", "state": state}
        )
        content = result.update["messages"][0].content
        self.assertIn("Papers have been reviewed but not yet approved", content)
