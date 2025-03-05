import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Import the tool function and related classes
from aiagents4pharma.talk2scholars.tools.zotero.zotero_read import (
    zotero_search_tool,
    ZoteroSearchInput,
)
from langgraph.types import Command

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
        # Setup Hydra mocks
        mock_hydra_compose.return_value = dummy_cfg
        # Simulate the context manager for hydra.initialize
        mock_hydra_init.return_value.__enter__.return_value = None

        # Create a fake Zotero client and simulate a successful items call
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

        # Patch the utility function to return a fake mapping for collection paths.
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

        # Verify that the result is a Command with expected update values.
        self.assertIsInstance(result, Command)
        update = result.update
        self.assertIn("zotero_read", update)
        self.assertIn("last_displayed_papers", update)
        self.assertIn("messages", update)

        filtered_papers = update["zotero_read"]
        self.assertIn("paper1", filtered_papers)
        self.assertIn("paper2", filtered_papers)
        # Check that the summary contains the query and number of papers
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
        # Setup Hydra mocks
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        # Fake Zotero client returns items when query is empty (fetching all items)
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
        fake_zot.items.return_value = fake_items  # Should be called with limit = dummy_cfg.tools.zotero_read.zotero.max_limit
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
        # Verify that Zotero.items was called with limit equal to max_limit (5 in dummy config)
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
        # Setup Hydra mocks
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_zot.items.return_value = []  # No items returned
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
        """
        Test that if all fetched items are filtered out (e.g., due to non-research item types),
        the tool raises a RuntimeError.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        # Create items with types that are excluded (e.g., "attachment")
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

        # Even if collections mapping exists, none of the items pass filtering
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
        """
        Test that if the Zotero API call raises an exception, the tool catches it
        and raises a RuntimeError.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        # Simulate an exception when calling items()
        fake_zot.items.side_effect = Exception("API error")
        mock_zotero_class.return_value = fake_zot

        # No need to set get_item_collections because the exception happens earlier.
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


if __name__ == "__main__":
    unittest.main()
