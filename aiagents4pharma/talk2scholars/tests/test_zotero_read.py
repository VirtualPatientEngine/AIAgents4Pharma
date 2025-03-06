"""
Unit tests for Zotero search tool in zotero_read.py.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
from langgraph.types import Command
from aiagents4pharma.talk2scholars.tools.zotero.zotero_read import (
    zotero_search_tool,
)


# Dummy Hydra configuration to be used in tests
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


class TestZoteroSearchTool(unittest.TestCase):
    """test for Zotero search tool"""

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_valid_query(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """test valid query"""
        # Setup Hydra mocks
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        # Create a fake Zotero client that returns two valid items
        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },
            {
                "data": {
                    "key": "paper2",
                    "title": "Paper 2",
                    "abstractNote": "Abstract 2",
                    "date": "2022",
                    "url": "http://example2.com",
                    "itemType": "conferencePaper",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot

        # Fake mapping for collection paths
        mock_get_item_collections.return_value = {
            "paper1": ["/Test Collection"],
            "paper2": ["/Test Collection"],
        }

        # Call the tool with a valid query using .run() with a dictionary input
        tool_call_id = "test_id_1"
        tool_input = {
            "query": "test",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        result = zotero_search_tool.run(tool_input)

        # Verify the Command update structure and contents
        self.assertIsInstance(result, Command)
        update = result.update
        self.assertIn("zotero_read", update)
        self.assertIn("last_displayed_papers", update)
        self.assertIn("messages", update)

        filtered_papers = update["zotero_read"]
        self.assertIn("paper1", filtered_papers)
        self.assertIn("paper2", filtered_papers)
        message_content = update["messages"][0].content
        self.assertIn("Query: test", message_content)
        self.assertIn("Number of papers found: 2", message_content)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_empty_query_fetch_all_items(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """test empty query fetches all items"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        tool_call_id = "test_id_2"
        tool_input = {
            "query": "  ",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        result = zotero_search_tool.run(tool_input)

        update = result.update
        filtered_papers = update["zotero_read"]
        self.assertIn("paper1", filtered_papers)
        fake_zot.items.assert_called_with(
            limit=dummy_cfg.tools.zotero_read.zotero.max_limit
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_no_items_returned(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """test no items returned from Zotero"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_zot.items.return_value = []
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_id_3"
        tool_input = {
            "query": "nonexistent",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_search_tool.run(tool_input)
        self.assertIn("No items returned from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_filtering_no_matching_papers(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """test no matching papers returned from Zotero"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "attachment",
                }
            },
            {
                "data": {
                    "key": "paper2",
                    "title": "Paper 2",
                    "abstractNote": "Abstract 2",
                    "date": "2022",
                    "url": "http://example2.com",
                    "itemType": "note",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {
            "paper1": ["/Test Collection"],
            "paper2": ["/Test Collection"],
        }

        tool_call_id = "test_id_4"
        tool_input = {
            "query": "test",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_search_tool.run(tool_input)
        self.assertIn("No matching papers returned from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_items_api_exception(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """test items API exception"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None
        mock_get_item_collections.return_value = {}

        fake_zot = MagicMock()
        fake_zot.items.side_effect = Exception("API error")
        mock_zotero_class.return_value = fake_zot

        tool_call_id = "test_id_5"
        tool_input = {
            "query": "test",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_search_tool.run(tool_input)
        self.assertIn("Failed to fetch items from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_missing_key_in_item(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """
        Test that an item with a valid 'data' structure but missing the 'key' field is skipped.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "title": "No Key Paper",
                    "abstractNote": "Abstract",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },  # missing key triggers line 136
            {
                "data": {
                    "key": "paper_valid",
                    "title": "Valid Paper",
                    "abstractNote": "Valid Abstract",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper_valid": ["/Test Collection"]}

        tool_call_id = "test_id_6"
        tool_input = {
            "query": "dummy",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        result = zotero_search_tool.run(tool_input)

        update = result.update
        filtered_papers = update["zotero_read"]
        self.assertIn("paper_valid", filtered_papers)
        self.assertEqual(len(filtered_papers), 1)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_item_not_dict(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """
        Test that if the items list contains an element that is not a dict, it is skipped.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        # Supply one item that is not a dict.
        fake_items = ["this is not a dict"]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        # Mapping doesn't matter here.
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_id_7"
        tool_input = {
            "query": "dummy",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_search_tool.run(tool_input)
        self.assertIn("No matching papers returned from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_read.hydra.initialize")
    def test_data_not_dict(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """
        Test that if an item has a 'data' field that is not a dict, it is skipped.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        # Supply one item whose "data" field is not a dict.
        fake_items = [{"data": "this is not a dict"}]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_id_8"
        tool_input = {
            "query": "dummy",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_search_tool.run(tool_input)
        self.assertIn("No matching papers returned from Zotero", str(context.exception))
