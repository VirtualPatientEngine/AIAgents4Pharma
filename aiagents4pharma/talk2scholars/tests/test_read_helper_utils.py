"""
Unit tests for Zotero read helper download branches.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper import (
    ZoteroSearchData,
)

# Dummy Hydra configuration for tests
dummy_zotero_read_config = SimpleNamespace(
    user_id="dummy_user",
    library_type="user",
    api_key="dummy_api_key",
    zotero=SimpleNamespace(
        max_limit=5,
        filter_item_types=["journalArticle", "conferencePaper"],
        filter_excluded_types=["attachment", "note"],
    ),
)
dummy_cfg = SimpleNamespace(tools=SimpleNamespace(zotero_read=dummy_zotero_read_config))


class TestReadHelperDownloadsFalse(unittest.TestCase):
    """Tests for read_helper download_pdfs=False branches."""

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_download_pdfs_false_branches(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Ensure attachment_key and filename are set when download_pdfs=False."""
        # Setup Hydra mocks
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        # Fake Zotero items: one paper with child PDF, one orphaned PDF
        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "P1",
                    "abstractNote": "A1",
                    "date": "2021",
                    "url": "u1",
                    "itemType": "journalArticle",
                }
            },
            {
                "data": {
                    "key": "attach2",
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                    "filename": "file2.pdf",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        # children for paper1
        fake_child = {
            "data": {
                "key": "attach1",
                "filename": "file1.pdf",
                "contentType": "application/pdf",
            }
        }

        def children_side_effect(key):
            return [fake_child] if key == "paper1" else []

        fake_zot.children.side_effect = children_side_effect
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/C1"], "attach2": ["/C2"]}

        # Instantiate with download_pdfs=False
        search = ZoteroSearchData(
            query="test",
            only_articles=False,
            limit=2,
            tool_call_id="id",
            download_pdfs=False,
        )
        search.process_search()
        data = search.get_search_results()["article_data"]

        # Regular paper1 should have attachment_key and filename, no pdf_url
        self.assertIn("paper1", data)
        self.assertEqual(data["paper1"]["attachment_key"], "attach1")
        self.assertEqual(data["paper1"]["filename"], "file1.pdf")
        self.assertNotIn("pdf_url", data["paper1"])

        # Orphan attach2 should have attachment_key and filename, no pdf_url
        self.assertIn("attach2", data)
        self.assertEqual(data["attach2"]["attachment_key"], "attach2")
        self.assertEqual(data["attach2"]["filename"], "file2.pdf")
        self.assertNotIn("pdf_url", data["attach2"])
