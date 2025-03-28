"""
Unit tests for Zotero write tool in zotero_write.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from aiagents4pharma.talk2scholars.tools.zotero.zotero_write import zotero_write

dummy_zotero_write_config = SimpleNamespace(
    user_id="dummy", library_type="user", api_key="dummy"
)
dummy_cfg = SimpleNamespace(
    tools=SimpleNamespace(zotero_write=dummy_zotero_write_config)
)


class TestZoteroSaveTool(unittest.TestCase):
    """Test class for Zotero save tool"""

    def setUp(self):
        """Patch Hydra and Zotero client globally"""
        self.hydra_init = patch(
            "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.hydra.initialize"
        ).start()
        self.hydra_compose = patch(
            "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.hydra.compose",
            return_value=dummy_cfg,
        ).start()
        self.zotero_class = patch(
            "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.zotero.Zotero"
        ).start()

        self.fake_zot = MagicMock()
        self.zotero_class.return_value = self.fake_zot

    def tearDown(self):
        """Stop all patches"""
        patch.stopall()

    def make_state(self, papers=None, approved=True, path="/Test Collection"):
        """Create a state dictionary with optional papers and approval info"""
        state = {}
        if approved:
            state["zotero_write_approval_status"] = {
                "approved": True,
                "collection_path": path,
            }
        if papers is not None:
            # When papers is provided as dict, use it directly.
            state["last_displayed_papers"] = (
                papers if isinstance(papers, dict) else "papers"
            )
            if isinstance(papers, dict):
                state["papers"] = papers
        return state

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.fetch_papers_for_save",
        return_value=None,
    )
    def test_no_papers_after_approval(self, mock_fetch):
        """Test when no fetched papers are found after approval"""
        with self.assertRaises(ValueError) as cm:
            zotero_write.run(
                {
                    "tool_call_id": "id",
                    "collection_path": "/Test Collection",
                    "state": self.make_state({}, True),
                }
            )
        self.assertIn("No fetched papers were found to save", str(cm.exception))
        mock_fetch.assert_called_once()

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.find_or_create_collection",
        return_value=None,
    )
    def test_invalid_collection(self, mock_find, mock_fetch):
        """Test when collection path is invalid"""
        self.fake_zot.collections.return_value = [
            {"key": "k1", "data": {"name": "Existing"}}
        ]
        # Provide a valid papers dict so we don't hit the no-papers error.
        state = self.make_state({"p1": {"Title": "X"}}, True)
        result = zotero_write.run(
            {
                "tool_call_id": "id",
                "collection_path": "/DoesNotExist",
                "state": state,
            }
        )
        # Remove outdated assertions and check for updated message content.
        content = result.update["messages"][0].content
        self.assertIn("does not exist in Zotero", content)
        self.assertIn("/DoesNotExist", content)
        self.assertIn("Existing", content)
        mock_fetch.return_value = {"p1": {"Title": "X"}}
        mock_find.return_value = None

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.find_or_create_collection",
        return_value="colKey",
    )
    def test_save_failure(self, mock_find, mock_fetch):
        """Test when Zotero save operation fails"""
        self.fake_zot.collections.return_value = [
            {"key": "colKey", "data": {"name": "Test Collection"}}
        ]
        self.fake_zot.create_items.side_effect = Exception("Creation error")
        state = self.make_state({"p1": {"Title": "X"}}, True)
        with self.assertRaises(RuntimeError) as cm:
            zotero_write.run(
                {
                    "tool_call_id": "id",
                    "collection_path": "/Test Collection",
                    "state": state,
                }
            )
        self.assertIn("Error saving papers to Zotero", str(cm.exception))
        mock_fetch.return_value = {"p1": {"Title": "X"}}
        mock_find.return_value = "colKey"

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.fetch_papers_for_save",
        return_value={"p1": {"Title": "X"}},
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.write_helper.find_or_create_collection",
        return_value="colKey",
    )
    def test_successful_save(self, mock_find, mock_fetch):
        """Test when Zotero save operation is successful"""
        self.fake_zot.collections.return_value = [
            {"key": "colKey", "data": {"name": "Test Collection"}}
        ]
        self.fake_zot.create_items.return_value = {
            "successful": {"0": {"key": "item1"}}
        }
        mock_fetch.return_value = {"p1": {"Title": "X"}}
        mock_find.return_value = "colKey"

        result = zotero_write.run(
            {
                "tool_call_id": "id",
                "collection_path": "/Test Collection",
                "state": self.make_state({"p1": {"Title": "X"}}, True),
            }
        )
        content = result.update["messages"][0].content
        self.assertIn("Save was successful", content)
        self.assertIn("Test Collection", content)
